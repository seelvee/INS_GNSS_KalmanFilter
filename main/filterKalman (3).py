import csv
from pathlib import Path

import numpy as np

omega_shu =  0.0012407 # rad/s
height = 167 # высота в метрах
e2 = 6.69437999014e-3
V_h = 0
sigma_pos_m = 3.0
sigma_vel_mps = 0.3
lat_msk = 55.36

gravity_nav = np.array([0.0, 0.0, -9.80665])
dt = 0.1
v_h = 0


def radii_of_curvature(lat_deg: float, height_m: float = 150.0):
    """
    Радиусы кривизны Земли для заданной широты.
    
    lat_deg  — широта в градусах
    height_m — высота, м
    
    Возвращает:
    rho_n — радиус меридиана, для пересчета широты в метры
    rho_e — радиус первого вертикала, для пересчета долготы в метры
    """

    A = 6378137.0 # экваториальный радиус Земли, м
    E2 = 6.69437999014e-3 # квадрат эксцентрис  
    lat = np.deg2rad(lat_deg)

    sin_lat = np.sin(np.deg2rad(lat))

    rho_n = A * (1 - E2) / (1 - E2 * sin_lat**2)**1.5 + height_m
    rho_e = A / np.sqrt(1 - E2 * sin_lat**2) + height_m

    return rho_n, rho_e


def lonlat_to_meters(lon: float, lat: float, height_m: float = 150.0):
    """
    Перевод координат lon/lat в локальные метры относительно точки lon0/lat0.

    lon, lat   — текущая точка в градусах
    lon0, lat0 — начальная точка в градусах

    Возвращает:
    east_m  — смещение на восток, м
    north_m — смещение на север, м
    """
    rho_n, rho_e = radii_of_curvature(lat, height_m)

    d_lat = np.deg2rad(lat)
    d_lon = np.deg2rad(lon)

    north_m = lat * rho_n
    east_m = lon * rho_e * np.cos(np.deg2rad(lat))

    return east_m, north_m


def meters_to_lonlat(east_m: float, north_m: float, lat0: float, height_m: float = 150.0):
    """
    Обратный перевод: из локальных метров в lon/lat.

    east_m, north_m — смещение в метрах
    lon0, lat0      — начальная точка в градусах

    Возвращает:
    lon, lat — координаты в градусах
    """
    rho_n, rho_e = radii_of_curvature(lat0, height_m)

    lat = north_m / rho_n
    lon = east_m / (rho_e * np.cos(np.deg2rad(lat0)))
    
    return lon, lat





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
        0.026603212389077466,
        0.018265213187570104,
        0.028088268862857684
    ])

    gyro_noise_std_rad_s = np.array([
        0.00179383401865371,
        0.0008979950425318393,
        0.0005319062935916511
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

    Q = G @ Qw @ G.T * dt**2

    return Q




# --------------------------------- Начало программы -------------------------------

path = "arduino_ins_gnss\\ins_gnss_20260506_211949.csv"
inputs = parse_csv(path)

# lat0 = inputs[1]["lat"] # широта начальная ФИ
# lon0 = inputs[1]["lon"] # долгота начальная ЛЯМБДА

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


P[0] = np.eye(n)
Fi[0] = np.eye(n)


lat_ins = np.zeros(N)
lon_ins = np.zeros(N)
lon_ins[0] = inputs[0]["lon"]
lat_ins[0] = inputs[0]["lat"]

lat0 = inputs[0]["lat"]
lon0 = inputs[0]["lon"]

for k in range(1, len(inputs)):

    lat = inputs[k]["lat"]
    lon = inputs[k]["lon"]
    a_x = inputs[k]["ax"]
    a_y = inputs[k]["ay"]
    a_z = inputs[k]["az"]
    g_x = np.deg2rad(inputs[k]["gx"])
    g_y = np.deg2rad(inputs[k]["gy"])
    g_z = np.deg2rad(inputs[k]["gz"])


    roll = np.deg2rad(inputs[k]["roll_deg"])
    pitch = np.deg2rad(inputs[k]["pitch_deg"])
    yaw = np.deg2rad(inputs[k]["yaw_deg"])
    

    sin_phi = np.sin(np.deg2rad(lat))
    rho1 = 6378137.0 * (1 - e2) / (1 - e2 * sin_phi**2)**(3 / 2) + height
    rho2 = 6378137.0 / np.sqrt(1 - e2 * sin_phi**2) + height

    a = np.array([a_x - X[k][10], a_y - X[k][11], a_z - X[k][12]]) # с учетом погрешностей
    omega = np.array([g_x, g_y, g_z])
    transf = body_to_normal_matrix(yaw, pitch, roll)
    a = transf @ a
    a = a + gravity_nav
    omega = transf @ omega
    
    n_e = a[1]
    n_H = a[2]
    n_N = a[0]

    if (inputs[k]["lat"] == inputs[k-1]["lat"] and inputs[k]["lon"] == inputs[k-1]["lon"] and k != 0):
        lat0 = lat_ins[k-1]
        lon0 = lon_ins[k-1]
    else:
        lat0 = inputs[k-1]["lat"] # фи
        lon0 = inputs[k-1]["lon"] # лямбда

    lat0 = np.deg2rad(lat0) / rho1 # перевод в метры
    lon0 = np.deg2rad(lon0) / (rho2 * np.cos(np.deg2rad(lat))) # перевод в метры
    lon_ins[0] = lon0
    lat_ins[0] = lat0

    # --------------- Часть кода которая интегрирует ускорение -----------------------
    v_n_ins[k] = v_n_ins[k-1] + n_N * dt # жесткая связь / метры
    v_e_ins[k] = v_e_ins[k-1] + n_e * dt # жесткая связь / метры
    lat_ins[k] = lat0 + (v_n_ins[k]) * dt # Метры
    lon_ins[k] = lon0 + (v_e_ins[k]) * dt #метры
    # --------------------------------------------------------------------------------


    # ВОТ ТУТ ПЕРЕСМОТРИ ВСЕ ЕЩЕ РАЗ !!!!!!!!!!!!!!!

    d_lon_m = lon_ins[k] - lon0 
    d_lat_m = lat_ins[k] - lat0

    Z[k] = np.array([
        d_lon_m,              # ошибка по East, м
        d_lat_m,              # ошибка по North, м
        0, # ошибка восточной скорости
        0  # ошибка северной скорости
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
    W = np.array([])
    Q[k] = build_Q_from_imu_noise(C, dt)
    
    
    sigma_lat = sigma_pos_m / rho1
    sigma_lon = sigma_pos_m / (rho2 * np.cos(np.deg2rad(lat_ins[k])))  

    R = np.diag([
        sigma_lon**2,
        sigma_lat**2,
        sigma_vel_mps**2,
        sigma_vel_mps**2,
    ])
    


    H[k] = np.array([
    [1 / (rho2 * np.cos(np.deg2rad(lat))), 0,          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0,                         1 / rho1,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-(V_h / rho2 + omega[1] * np.tan(np.deg2rad(lat))), -omega[2], 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [-V_h / rho1,               0,          0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    G[k] = build_G(C)
    Fi[k] = np.eye(13) + F[k]*dt
    Ge[k] = G[k] * dt

    # фильтр калмана ----------------
    S[k] = (Fi[k-1] @ P[k-1] @ np.transpose(Fi[k-1])) + (Q[k-1])
    K[k] = S[k] @ np.transpose(H[k]) @ np.linalg.inv( H[k] @ S[k] @ np.transpose(H[k]) + R)
    P[k] = (np.eye(13) - K[k] @ H[k]) @ S[k]

    X[k] = Fi[k-1] @ X[k-1] + K[k] @ (Z[k] - H[k] @ Fi[k-1] @ X[k-1] )

    lon_kalm[k] = lon_ins[k] - X[k][0] / (rho2 * np.cos(np.deg2rad(lat_ins[k])))
    lat_kalm[k] = lat_ins[k] - X[k][1] / rho1

    v_e_kalm[k] = v_e_ins[k] - X[k][2]
    v_n_kalm[k] = v_n_ins[k] - X[k][3]

    v_e_ins[k] = v_e_ins[k] - X[k][2]
    v_n_ins[k] = v_n_ins[k] - X[k][3]








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

gps_points = make_points(gps_lat, gps_lon)
# gps_points = np.zeros((len(gps_lat), 2))



for k in range(len(lat_ins)):
    lon_ins[k], lat_ins[k] = meters_to_lonlat(lat_ins[k], lon_ins[k], lat_msk, height)

ins_points = make_points(lat_ins, lon_ins)
kalman_points = make_points(lat_kalm, lon_kalm)

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