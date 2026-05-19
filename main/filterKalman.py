import csv
from pathlib import Path
import numpy as np


path = "data\\multi_1778262478.318379.csv"

omega_shu =  0.0012407 # rad/s частота шуллера
height = 167 # высота в метрах
e2 = 6.69437999014e-3 # квадратичный эксцентриситет земного эллипсоида
V_h = 0 # горизонтальная скорость, м/с (можно оценить по данным GNSS или ИНС)
sigma_pos_m = 2.10 # стандартное отклонение GNSS-позиции, м
sigma_vel_mps = 0.110 # стандартное отклонение GNSS-скорости, м/с
lat_msk = 55.36 # приблизительная широта Москвы, градусы


gravity_nav = np.array([0.0, 0.0, -9.80665]) # вектор ускорения свободного падения в навигационной СК (North-East-Down)
dt = 0.01 # шаг дискретизации, с (можно оценить по данным GNSS или ИНС)
v_h = 0 # горизонтальная скорость, м/с (можно оценить по данным GNSS или ИНС)

#----------------------------- Функции -----------------------------

def radii_of_curvature(lat_deg: float = lat_msk, height_m: float = 150.0):
    """
    Радиусы кривизны Земли для заданной широты.

    lat_deg  — широта в градусах
    height_m — высота, м

    Возвращает:
    rho_n — радиус меридиана, м
    rho_e — радиус первого вертикала, м
    """

    A = 6378137.0 # радиус земного эллипсоида
    E2 = 6.69437999014e-3 # квадратичный эксцентриситет земного эллипсоида

    lat_rad = np.deg2rad(lat_deg) # широта в радианах
    sin_lat = np.sin(lat_rad)

    rho_n = A * (1 - E2) / (1 - E2 * sin_lat**2)**1.5 + height_m
    rho_e = A / np.sqrt(1 - E2 * sin_lat**2) + height_m

    return rho_n, rho_e


def lat_deg_to_meters(delta_lat_deg: float, lat_ref_deg: float = lat_msk, height_m: float = 150.0):
    """
    Смещение по широте: градусы -> метры.

    delta_lat_deg — изменение широты, градусы
    lat_ref_deg   — опорная широта, градусы
    """

    rho_n, _ = radii_of_curvature(lat_ref_deg, height_m)

    delta_lat_rad = np.deg2rad(delta_lat_deg)
    dy_m = delta_lat_rad * rho_n

    return dy_m


def lat_meters_to_deg(dy_m: float, lat_ref_deg: float = lat_msk, height_m: float = 150.0):
    """
    Смещение по широте: метры -> градусы.

    dy_m        — смещение на север/юг, м
    lat_ref_deg — опорная широта, градусы
    """

    rho_n, _ = radii_of_curvature(lat_ref_deg, height_m)

    delta_lat_rad = dy_m / rho_n
    delta_lat_deg = np.rad2deg(delta_lat_rad)

    return delta_lat_deg


def lon_deg_to_meters(delta_lon_deg: float, lat_ref_deg: float = lat_msk, height_m: float = 150.0):
    """
    Смещение по долготе: градусы -> метры.

    delta_lon_deg — изменение долготы, градусы
    lat_ref_deg   — опорная широта, градусы
    """

    _, rho_e = radii_of_curvature(lat_ref_deg, height_m)

    lat_rad = np.deg2rad(lat_ref_deg)
    delta_lon_rad = np.deg2rad(delta_lon_deg)

    dx_m = delta_lon_rad * rho_e * np.cos(lat_rad)

    return dx_m


def lon_meters_to_deg(dx_m: float, lat_ref_deg: float = lat_msk, height_m: float = 150.0):
    """
    Смещение по долготе: метры -> градусы.

    dx_m        — смещение на восток/запад, м
    lat_ref_deg — опорная широта, градусы
    """

    _, rho_e = radii_of_curvature(lat_ref_deg, height_m)

    lat_rad = np.deg2rad(lat_ref_deg)

    delta_lon_rad = dx_m / (rho_e * np.cos(lat_rad))
    delta_lon_deg = np.rad2deg(delta_lon_rad)

    return delta_lon_deg

def build_G(C):

    G = np.zeros((13, 6))

    # Строки 3–4: влияние шумов акселерометров через C
    G[2:4, 3:6] = C[0:2, :]

    # Строки 5–7: влияние шумов ДУС/гироскопов через C
    G[4:7, 0:3] = C

    return G

def _to_float(value: str) -> float:
    """Convert CSV token to float, keeping invalid values as np.nan."""
    token = value.strip().lower()
    if token in {"", "nan", "none"}:
        return np.nan
    try:
        return float(token)
    except ValueError:
        return np.nan
    
def parse_csv(csv_path: str | Path) -> list[dict[str, float]]:
    """
    Read telemetry CSV and return rows as dictionaries.

    File format:
    - comment lines start with '#'
    - one header line
    - numeric values (including 'nan')
    """
    rows: list[dict[str, float]] = []

    with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file)

        header: list[str] | None = None
        for raw_row in reader:
            if not raw_row:
                continue
            if raw_row[0].strip().startswith("#"):
                continue

            if header is None:
                header = [col.strip() for col in raw_row]
                continue

            row = {header[i]: _to_float(raw_row[i]) for i in range(min(len(header), len(raw_row)))}
            rows.append(row)

    return rows

def body_to_normal_matrix(psi, theta, gamma): #  получится NEH
    """
    Матрица C_b^n: из связанной СК в нормальную СК.
    
    psi   — курс, yaw
    theta — тангаж, pitch
    gamma — крен, roll
    
    Все углы в радианах.
    """

    c_psi = np.cos(psi)
    s_psi = np.sin(psi)

    c_theta = np.cos(theta)
    s_theta = np.sin(theta)

    c_gamma = np.cos(gamma)
    s_gamma = np.sin(gamma)

    C = np.array([
        [
            c_theta * c_psi,
            s_gamma * s_theta * c_psi - c_gamma * s_psi,
            c_gamma * s_theta * c_psi + s_gamma * s_psi
        ],
        [
            c_theta * s_psi,
            s_gamma * s_theta * s_psi + c_gamma * c_psi,
            c_gamma * s_theta * s_psi - s_gamma * c_psi
        ],
        [
            -s_theta,
            s_gamma * c_theta,
            c_gamma * c_theta
        ]
    ])

    return C

def build_Q_from_imu_noise(C, dt):
    accel_noise_std_mps2 = np.array([
        0.26603212389077466,
        0.18265213187570104,
        0.28088268862857684
    ])

    gyro_noise_std_rad_s = np.array([
        0.0179383401865371,
        0.08979950425318393,
        0.053190629359165
    ])

    Qw = np.diag([
        gyro_noise_std_rad_s[0]**2,
        gyro_noise_std_rad_s[1]**2,
        gyro_noise_std_rad_s[2]**2,

        accel_noise_std_mps2[0]**2,
        accel_noise_std_mps2[1]**2,
        accel_noise_std_mps2[2]**2,
    ])

    G = build_G(C)

    Q = G @ Qw @ G.T * dt

    return Q



#----------------------------- Параметры и переменные фильтра Калмана -----------------------------
result = np.array(np.array)

x1              = 0 # ошибка долготы в метрах
x2              = 0 # ошибка широты в метрах    
x3              = 0 # ошибка восточной скорости
x4              = 0 # ошибка северной скорости
alpha           = 0 # погрешность ориентации дрейфа ДУС
beta            = 0 # погрешность ориентации дрейфа ДУС
gamma           = 0 # погрешность ориентации дрейфа ДУС
delta_omega_x   = 0 # проекция постоянных погрешностей ДУС в связанной
delta_omega_y   = 0 # проекция постоянных погрешностей ДУС в связанной
delta_omega_z   = 0 # проекция постоянных погрешностей ДУС в связанной
delta_n_x       = 0 # постоянные составляющие ошибок акселерометров в проекции на оси связанной СК
delta_n_y       = 0 # постоянные составляющие ошибок акселерометров в проекции на оси связанной СК
delta_n_z       = 0 # постоянные составляющие ошибок акселерометров в проекции на оси связанной СК


X = np.array(np.array([
    x1,
    x2,
    x3,
    x4,
    alpha,
    beta,
    gamma,
    delta_omega_x,
    delta_omega_y,
    delta_omega_z,
    delta_n_x,
    delta_n_y,
    delta_n_z
], dtype=float))


# -------------------------------- Начало программы -------------------------------

inputs = parse_csv(path)

v_n_ins = np.zeros(len(inputs))
v_e_ins = np.zeros(len(inputs))

N = len(inputs)

n = 13
m = 4

F  = np.zeros((N, n, n))
Fi = np.zeros((N, n, n))
Q  = np.zeros((N, n, n))
P  = np.zeros((N, n, n))

G  = np.zeros((N, n, 6))
Ge = np.zeros((N, n, 6))

H = np.zeros((N, m, n))
S = np.zeros((N, 13, 13))
K = np.zeros((N, n, m))
X = np.zeros((N,n))
Z = np.zeros((N,4))
lon_kalm = np.zeros(N)
lat_kalm = np.zeros(N)
v_e_kalm = np.zeros(N)
v_n_kalm = np.zeros(N)


P[0] = np.diag([
    0.10**2,     # ошибка долготы/востока, м^2
    01.10**2,     # ошибка широты/севера, м^2
    1.0**2,       # ошибка скорости по востоку, (м/с)^2
    1.0**2,       # ошибка скорости по северу, (м/с)^2
    np.deg2rad(5.0)**2,   # неопределенность крена, рад^2
    np.deg2rad(5.0)**2,   # неопределенность тангажа, рад^2
    np.deg2rad(10.0)**2,  # неопределенность курсового дрейфа, рад^2
    0.01**2,      # неопределенность смещения гироскопа x, (рад/с)^2
    0.1**2,      # uncertainty gyro bias y
    0.01**2,      # uncertainty gyro bias z
    0.2**2,       # uncertainty accel bias x, (м/с^2)^2
    0.2**2,       # uncertainty accel bias y
    0.2**2        # uncertainty accel bias z
])

Fi[0] = np.eye(n)


lat_ins = np.zeros(N)
lon_ins = np.zeros(N)
lon_ins[0] = lon_deg_to_meters(inputs[0]["lon"])
lat_ins[0] = lat_deg_to_meters(inputs[0]["lat"])


#----------------------------- Основной цикл по данным -----------------------------
for k in range(1, len(inputs)):

    # Извлекаем измерения из строки CSV.
    lat = inputs[k]["lat"]
    lon = inputs[k]["lon"]
    dt_k = dt
    a_x = inputs[k]["gFx"] * 9.80665
    a_y = inputs[k]["gFy"] * 9.80665
    a_z = inputs[k]["gFz"] * 9.80665
    g_x = np.deg2rad(inputs[k]["wx"])
    g_y = np.deg2rad(inputs[k]["wy"])
    g_z = np.deg2rad(inputs[k]["wz"])

    roll = np.deg2rad(inputs[k]["roll"])
    pitch = np.deg2rad(inputs[k]["pitch"])
    yaw = np.deg2rad(inputs[k]["yaw"])
    gnss_speed = inputs[k]["speed"]
    # Если имеются GNSS скорости, проецируем их на оси для сравнения с ИНС.
    v_gnss_e = gnss_speed * np.sin(yaw)
    v_gnss_n = gnss_speed * np.cos(yaw)


    a = np.array([a_x - X[k - 1][10], a_y - X[k - 1][11], a_z - X[k - 1][12]]) # с учетом погрешностей
    omega = np.array([g_x, g_y, g_z])
    transf = body_to_normal_matrix(yaw, pitch, roll)
    a = transf @ a
    a = a - gravity_nav
    omega = transf @ omega
    
    n_e = a[1] 
    n_H = a[2]
    n_N = a[0]

    # --------------- Часть кода которая интегрирует ускорение -----------------------
    v_n_ins[k] = v_n_ins[k-1] + n_N * dt # жесткая связь / метры
    v_e_ins[k] = v_e_ins[k-1] + n_e * dt # жесткая связь / метры
    lat_ins[k] = lat_ins[k-1] + (v_n_ins[k]) * dt # Метры
    lon_ins[k] = lon_ins[k-1] + (v_e_ins[k]) * dt #метры
    # --------------------------------------------------------------------------------


    d_lon_m = lon_deg_to_meters(lon_meters_to_deg(lon_ins[k]) - lon)
    d_lat_m = lat_deg_to_meters(lat_meters_to_deg(lat_ins[k]) - lat)

    # Z[k] = np.array([
    #     d_lon_m,              # ошибка по East, м
    #     d_lat_m,              # ошибка по North, м
    #     0.000, # ошибка восточной скорости
    #     0.000  # ошибка северной скорости
    # ])

    Z[k] = np.array([
        lon_deg_to_meters((lon_meters_to_deg(lon_ins[k]) - lon)) * 1,
        lat_deg_to_meters((lat_meters_to_deg(lat_ins[k]) - lat)) * 1,
        v_e_ins[k] - v_gnss_e,
        v_n_ins[k] - v_gnss_n,
    ])


    C = transf

    F[k] = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [-omega_shu**2, 0, 0, 0, n_H, 0, 0, 0, 0, C[0, 0], C[0, 1], C[0, 2], 0],
    [0, -omega_shu**2, 0, 0, -n_H, 0, n_e, 0, 0, C[1, 0], C[1, 1], C[1, 2], 0],
    [0, 0, 0, 0, 0, 0, 0, C[0, 0], C[0, 1], C[0, 2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, C[1, 0], C[1, 1], C[1, 2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, C[2, 0], C[2, 1], C[2, 2], 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       0,       0,       0],
    ])

    Q[k] = build_Q_from_imu_noise(C, dt) # она зависит от ориентации аппарата
    R = np.diag([
        sigma_pos_m**2,
        sigma_pos_m**2,
        sigma_vel_mps**2,
        sigma_vel_mps**2,
    ])

    H[k] = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    G[k] = build_G(C)
    Q[k] = Q[k]
    Fi[k] = np.eye(13) + F[k] * dt
    Ge[k] = G[k] * dt

    

    # фильтр калмана ----------------
    x_pred = Fi[k] @ X[k - 1]
    S[k] = (Fi[k-1] @ P[k-1] @ np.transpose(Fi[k-1])) + (Q[k-1])
    K[k] = S[k] @ np.transpose(H[k]) @ np.linalg.inv( H[k] @ S[k] @ np.transpose(H[k]) + R)
    P[k] = (np.eye(13) - K[k] @ H[k]) @ S[k]
    X[k] = x_pred + K[k] @ (Z[k] - H[k] @ x_pred) 

    
    # if inputs[k]["lat"] == inputs[k-1]["lat"]:
    #     X[k] = x_pred
    # else: 
    #     X[k] = x_pred + K[k] @ (Z[k] - H[k] @ x_pred) 


    lon_kalm[k] = lon_ins[k] - X[k][0]
    lat_kalm[k] = lat_ins[k] - X[k][1]
    v_e_kalm[k] = v_e_ins[k] - X[k][2]
    v_n_kalm[k] = v_n_ins[k] - X[k][3]

    v_e_ins[k] = v_e_ins[k] - X[k][2]
    v_n_ins[k] = v_n_ins[k] - X[k][3]
    lon_ins[k] = lon_ins[k] - X[k][0]
    lat_ins[k] = lat_ins[k] - X[k][1]








# --------------------- вывод результатов на карту --------------------
import folium
import webbrowser


def make_points(lat_array, lon_array):
    points = []
    for lat, lon in zip(lat_array, lon_array):
        if np.isfinite(lat) and np.isfinite(lon):
            if abs(lat) > 0.000001 and abs(lon) > 0.000001:
                points.append([lat, lon])
    return points


# GNSS-координаты из CSV

gps_lat = np.array([row["lat"] for row in inputs])
gps_lon = np.array([row["lon"] for row in inputs])


for k in range(len(lat_ins)):
    lon_ins[k], lat_ins[k] = lon_meters_to_deg(lon_ins[k]), lat_meters_to_deg(lat_ins[k])

gps_points = make_points(gps_lat, gps_lon)
ins_points = make_points(lat_ins, lon_ins)
kalman_points = make_points(lat_kalm, lon_kalm)

# ins_points = np.zeros((len(gps_lat), 2))
# kalman_points = np.zeros((len(gps_lat), 2))
# gps_points = np.zeros((len(gps_lat), 2))

# Центр карты — первая нормальная GNSS-точка
m = folium.Map(
    location=gps_points[0],
    zoom_start=18,
    tiles="OpenStreetMap"
)

# Траектория GNSS
folium.PolyLine(
    gps_points,
    tooltip="GNSS",
    color="blue",
    weight=5
).add_to(m)

# Траектория чистой ИНС
if len(ins_points) > 1:
    folium.PolyLine(
        ins_points,
        tooltip="INS",
        color="red",
        weight=3
    ).add_to(m)

# Траектория после Калмана
if len(kalman_points) > 1:
    folium.PolyLine(
        kalman_points,
        tooltip="Kalman",
        color="green",
        weight=4
    ).add_to(m)

# Старт и финиш
folium.Marker(
    gps_points[0],
    tooltip="Start",
    popup="Начало"
).add_to(m)

folium.Marker(
    gps_points[-1],
    tooltip="Finish",
    popup="Конец"
).add_to(m)

# Сохранение карты
map_path = Path("trajectory_map.html")
m.save(map_path)

# Автоматически открыть в браузере
webbrowser.open(map_path.resolve().as_uri())

import matplotlib.pyplot as plt

state_names = [
    "lon/east error",
    "lat/north error",
    "v_e error",
    "v_n error",
    "alpha",
    "beta",
    "gamma",
    "gyro bias x",
    "gyro bias y",
    "gyro bias z",
    "accel bias x",
    "accel bias y",
    "accel bias z",
]

plt.figure(figsize=(12, 7))

for i in range(n):
    plt.plot(P[:, i, i], label=state_names[i])

plt.xlabel("Step k")
plt.ylabel("Variance")
plt.title("Diagonal elements of covariance matrix P")
plt.grid(True)
plt.legend()
plt.show()


plt.figure(figsize=(12, 7))
for i in range(n):
    plt.plot(X[:, i], label=state_names[i])

plt.xlabel("Step k")
plt.ylabel("Variance")
plt.title("X")
plt.grid(True)
plt.legend()
plt.show()


gps_e = np.array([lon_deg_to_meters(row["lon"] - inputs[0]["lon"]) for row in inputs])
gps_n = np.array([lat_deg_to_meters(row["lat"] - inputs[0]["lat"]) for row in inputs])

plt.figure(figsize=(8, 8))
plt.plot(gps_e, gps_n, label="GNSS")
plt.plot(lon_ins, lat_ins, label="INS")
plt.plot(lon_kalm, lat_kalm, label="Kalman")
plt.axis("equal")
plt.grid()
plt.legend()
plt.xlabel("East, m")
plt.ylabel("North, m")
plt.title("Trajectory in local EN frame")
plt.show()

err_ins_e = lon_ins - gps_e
err_ins_n = lat_ins - gps_n

err_kal_e = lon_kalm - lon
err_kal_n = lat_kalm - lat

plt.figure(figsize=(12, 6))
plt.plot(X[:, 2], label="INS - GNSS East velocity")
plt.grid()
plt.legend()
plt.title("East velocity error")
plt.ylabel("Error, m/s")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(X[:, 3], label="INS - GNSS North velocity")
plt.grid()
plt.legend()
plt.title("North velocity error")
plt.ylabel("Error, m/s")
plt.show()

# Показать стандартного отклонения по позициям и скоростям
plt.figure(figsize=(12, 6))
plt.plot(np.sqrt(P[:, 0, 0]), label="sigma east pos")
plt.plot(np.sqrt(P[:, 1, 1]), label="sigma north pos")
plt.grid()
plt.legend()
plt.title("Position uncertainty sqrt(P)")
plt.ylabel("m")
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(np.sqrt(P[:, 2, 2]), label="sigma v_e")
plt.plot(np.sqrt(P[:, 3, 3]), label="sigma v_n")
plt.grid()
plt.legend()
plt.title("Velocity uncertainty sqrt(P)")
plt.ylabel("m/s")
plt.show()

