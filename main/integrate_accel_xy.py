import csv
from pathlib import Path
import numpy as np

# Скрипт читает лог IMU + GNSS, интегрирует ускорения в плоскости XY,
# строит локальную навигационную траекторию, использует простую модель
# Калман-фильтра для коррекции ошибок и визуализирует траектории.

omega_shu = 0.0012407  # частота шуллера
height = 167  # высота над уровнем моря, м 
e2 = 6.69437999014e-3  # квадратичный эксцентриситет земного эллипсоида
sigma_pos_m = 2.10  # стандартное отклонение GNSS-позиции, м
sigma_vel_mps = 0.110  # стандартное отклонение GNSS-скорости, м/с
lat_msk = 55.36  # приблизительная широта Москвы, градусы


gravity_nav = np.array([0.0, 0.0, -9.80665])  # гравитация в навигационной системе координат [N, E, D]
dt = 0.01  # базовый шаг интегрирования, секунды
max_gnss_pos_residual_m = 1500.0  # максимальный допустимый остаток GNSS-позиции, м
min_dt_s = 0.005  # минимальный интервал времени для корректных данных, с
max_dt_s = 0.2  # максимальный интервал времени для корректных данных, с
ins_start_blend_sec = 15.0  # время мягкого смешивания ИНС и GNSS в начале, с
ins_static_speed_threshold_mps = 0.6  
ins_bias_calib_max_samples = 250  # число первых выборок для калибровки постоянного смещения акселерометров
ins_accel_lpf_alpha = 0.15  # коэффициент фильтрации ускорения в навигационной системе
ins_hard_sync_interval_sec = 1000000.0  # интервал жесткой привязки ИНС к GNSS, с



def radii_of_curvature(lat_deg: float = lat_msk, height_m: float = 150.0):
    """
    Радиусы кривизны Земли для заданной широты.

    lat_deg  — широта в градусах
    height_m — высота, м

    Возвращает:
    rho_n — радиус меридиана, м
    rho_e — радиус первого вертикала, м
    """

    A = 6378137.0
    E2 = 6.69437999014e-3

    lat_rad = np.deg2rad(lat_deg)
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



# Вектор состояния X содержит ошибки и дрейфы системы:
# [0] lon/east error,
# [1] lat/north error,
# [2] east velocity error,
# [3] north velocity error,
# [4] alpha   — ошибка ориентации по крену (roll drift),
# [5] beta    — ошибка ориентации по тангажу (pitch drift),
# [6] gamma   — ошибка ориентации по курсу (yaw drift),
# [7..9] gyro biases in body frame,
# [10..12] accel biases in body frame.
x1 = 0  # ошибка долготы в метрах
x2 = 0  # ошибка широты в метрах
x3 = 0  # ошибка восточной скорости
x4 = 0  # ошибка северной скорости
alpha = 0  # погрешность ориентации дрейфа ДУС
beta = 0   # погрешность ориентации дрейфа ДУС
gamma = 0  # погрешность ориентации дрейфа ДУС
delta_omega_x = 0  # проекция постоянных погрешностей гироскопа на связанное тело
delta_omega_y = 0  # проекция постоянных погрешностей гироскопа на связанное тело
delta_omega_z = 0  # проекция постоянных погрешностей гироскопа на связанное тело
delta_n_x = 0  # постоянное смещение акселерометра по оси x
delta_n_y = 0  # постоянное смещение акселерометра по оси y
delta_n_z = 0  # постоянное смещение акселерометра по оси z

X = np.array([
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
], dtype=float)


def geodetic_to_local_m(lat_deg: float, lon_deg: float, lat_ref_deg: float, lon_ref_deg: float):
    """Convert geodetic coordinates to local EN displacements in meters."""
    d_lat = lat_deg - lat_ref_deg
    d_lon = lon_deg - lon_ref_deg
    north_m = lat_deg_to_meters(d_lat, lat_ref_deg=lat_ref_deg, height_m=height)
    east_m = lon_deg_to_meters(d_lon, lat_ref_deg=lat_ref_deg, height_m=height)
    return east_m, north_m


def local_m_to_geodetic(east_m: float, north_m: float, lat_ref_deg: float, lon_ref_deg: float):
    """Convert local EN displacements in meters back to geodetic coordinates."""
    lat_deg = lat_ref_deg + lat_meters_to_deg(north_m, lat_ref_deg=lat_ref_deg, height_m=height)
    lon_deg = lon_ref_deg + lon_meters_to_deg(east_m, lat_ref_deg=lat_ref_deg, height_m=height)
    return lat_deg, lon_deg


def build_G(C):
    """Build the process noise mapping matrix for the error state dynamics."""

    G = np.zeros((13, 6))

    # Акселерометрные шумы влияют на ошибки скоростей через элементы C.
    G[2:4, 3:6] = C[0:2, :]

    # Шумы гироскопа влияют на ошибки углов ориентации.
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


def sample_dt_s(curr_row: dict[str, float], prev_row: dict[str, float], default_dt_s: float = dt) -> float:
    """Return time increment from GNSS time stamps or fallback to default step."""
    t_curr = curr_row.get("time_ms", np.nan)
    t_prev = prev_row.get("time_ms", np.nan)
    if np.isfinite(t_curr) and np.isfinite(t_prev):
        dt_s = (t_curr - t_prev) * 1e-3
        if min_dt_s <= dt_s <= max_dt_s:
            return float(dt_s)
    return default_dt_s


def body_to_normal_matrix(psi, theta, gamma):
    """Compute the direction cosine matrix from body frame to navigation frame.

    The navigation frame is assumed to be North-East-Down (NED).
    Angles are given as yaw (psi), pitch (theta), roll (gamma).
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
    """Build the process noise covariance Q from IMU noise standard deviations."""
    # Оценочные стандартные отклонения шума акселерометра в м/с^2.
    accel_noise_std_mps2 = np.array([
        0.126603212389077466,
        0.118265213187570104,
        0.128088268862857684
    ])

    # Оценочные стандартные отклонения шума гироскопа в рад/с.
    gyro_noise_std_rad_s = np.array([
        0.0179383401865371,
        0.08979950425318393,
        0.05319062935916511
    ])

    Qw = np.diag([
        gyro_noise_std_rad_s[0]**2,
        gyro_noise_std_rad_s[1]**2,
        gyro_noise_std_rad_s[2]**2,

        accel_noise_std_mps2[0]**2,
        accel_noise_std_mps2[1]**2,
        accel_noise_std_mps2[2]**2,
    ])

    # G показывает, как шума IMU влияет на состояние, а умножение G*Qw*G.T дает ковариацию
    # процесса в пространстве состояния. Умножаем на dt для приведения к дискретной модели.
    G = build_G(C)
    Q = G @ Qw @ G.T * dt

    return Q




# --------------------------------- Начало программы -------------------------------
# Настройка исходных данных и резервирование массивов

path = "arduino_ins_gnss\\multi_1778262478.318379.csv"
inputs = parse_csv(path)

if not inputs:
    raise ValueError("CSV is empty or has no data rows.")

# Массивы для оценок ИНС по осям N/E
v_n_ins = np.zeros(len(inputs))
v_e_ins = np.zeros(len(inputs))

N = len(inputs)

# Размеры состояния и измерений
n = 13  # размер вектора состояния X
m = 4   # количество измерений: [d_lon, d_lat, d_ve, d_vn]

F = np.zeros((N, n, n))  # матрица Якоби линейной модели для каждого шага
Fi = np.zeros((N, n, n))  # дискретизированная матрица перехода состояния
Q = np.zeros((N, n, n))  # ковариация шума процесса
P = np.zeros((N, n, n))  # ковариационная матрица ошибок состояния

G = np.zeros((N, n, 6))  # чувствительность состояния к шума IMU
Ge = np.zeros((N, n, 6))  # дискретная матрица влияния шума

H = np.zeros((N, m, n))  # матрица наблюдения для измерений GNSS
S = np.zeros((N, 13, 13))
K = np.zeros((N, n, m))
X = np.zeros((N, n))  # история оценок вектора состояния
Z = np.zeros((N, 4))  # история измерений ошибок
lon_kalm = np.zeros(N)
lat_kalm = np.zeros(N)
v_e_kalm = np.zeros(N)
v_n_kalm = np.zeros(N)

# -------------------------
lat_real_ins = np.zeros(len(inputs))
lon_real_ins = np.zeros(len(inputs))

# Начальные геодезические координаты опорной точки
lat_ref_deg = inputs[0]["lat"]
lon_ref_deg = inputs[0]["lon"]

# Интегрируемая «чистая» ИНС-траектория без коррекций фильтра
v_n_pure = np.zeros(len(inputs))
v_e_pure = np.zeros(len(inputs))

# Позиция в метрах в локальной системе East-North-Up
east_m = 0.0
north_m = 0.0

lat_real_ins[0] = lat_ref_deg
lon_real_ins[0] = lon_ref_deg
# ----------------------------



# Начальная ковариация ошибок состояния. Вектор ошибки X стартует с заданной
# неопределенностью для позиции, скорости, углов ориентации и IMU-смещения.
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
Fi[0] = np.eye(n)  # начальная матрица перехода как единичная при k=0


lat_ins = np.zeros(N)
lon_ins = np.zeros(N)


lat_ref_deg = inputs[0]["lat"]
lon_ref_deg = inputs[0]["lon"]
v_n_ins[0] = 0
v_e_ins[0] = 0


# lat_ref_deg = inputs[0]["lat"]
# lon_ref_deg = inputs[0]["lon"]
# lon_ins[0], lat_ins[0] = geodetic_to_local_m(lat_ref_deg, lon_ref_deg, lat_ref_deg, lon_ref_deg)
# if np.isfinite(inputs[0].get("gnss_vn_mps", np.nan)):
#     v_n_ins[0] = inputs[0]["gnss_vn_mps"]
# if np.isfinite(inputs[0].get("gnss_ve_mps", np.nan)):
#     v_e_ins[0] = inputs[0]["gnss_ve_mps"]

accel_bias_nav = np.zeros(3)
accel_bias_count = 0
a_nav_lpf_prev = np.zeros(3)
last_ins_hard_sync_sec = 0.0



for k in range(1, len(inputs)):
    # Продолжаем предыдущее состояние, чтобы иметь устойчивость при пропусках данных.
    lat_ins[k] = lat_ins[k - 1]
    lon_ins[k] = lon_ins[k - 1]
    v_n_ins[k] = v_n_ins[k - 1]
    v_e_ins[k] = v_e_ins[k - 1]
    X[k] = X[k - 1]
    P[k] = P[k - 1]
    Fi[k] = Fi[k - 1]
    Q[k] = Q[k - 1]
    lon_kalm[k] = lon_kalm[k - 1]
    lat_kalm[k] = lat_kalm[k - 1]
    v_e_kalm[k] = v_e_kalm[k - 1]
    v_n_kalm[k] = v_n_kalm[k - 1]

    # Извлекаем измерения из строки CSV.
    lat = inputs[k]["lat"]
    lon = inputs[k]["lon"]
    dt_k = sample_dt_s(inputs[k], inputs[k - 1], dt)
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
    
    
    
    if not np.all(np.isfinite([lat, lon, a_x, a_y, a_z, g_x, g_y, g_z, roll, pitch, yaw])):
        continue

    # Учитываем текущие смещения акселерометров из состояния X.
    a = np.array([a_x - X[k - 1][10], a_y - X[k - 1][11], a_z - X[k - 1][12]])
    omega = np.array([g_x, g_y, g_z])
    transf = body_to_normal_matrix(yaw, pitch, roll)

    # Гравитация в связанной системе координат.
    # Переводим гравитационный вектор в тело и вычитаем из измерений акселерометра.
    gravity_body = transf.T @ gravity_nav
    a_lin_body = a - gravity_body
    a = transf @ a_lin_body

    # Применяем фильтр нижних частот к ускорению, чтобы сгладить выбросы.
    a_nav_lpf_prev = (1.0 - ins_accel_lpf_alpha) * a_nav_lpf_prev + ins_accel_lpf_alpha * a
    a = a_nav_lpf_prev.copy()

    # Оценка остаточного смещения ускорений при малой скорости.
    if (
        accel_bias_count < ins_bias_calib_max_samples
        and np.isfinite(gnss_speed)
        and gnss_speed <= ins_static_speed_threshold_mps
    ):
        accel_bias_nav = (accel_bias_nav * accel_bias_count + a) / (accel_bias_count + 1)
        accel_bias_count += 1

    a = a - accel_bias_nav
    omega = transf @ omega

    # Компоненты ускорения в навигационной системе.
    n_e = a[1]
    n_H = a[2]
    n_N = a[0]

    # Интегрируем ускорение, получая скорости ИНС.
    v_n_ins[k] = v_n_ins[k - 1] + n_N * dt_k
    v_e_ins[k] = v_e_ins[k - 1] + n_e * dt_k

    # Мягкое начальное смешение ИНС-скорости с GNSS-значениями.
    t_ms = inputs[k].get("time_ms", np.nan)
    t0_ms = inputs[0].get("time_ms", np.nan)
    t_sec = ((t_ms - t0_ms) * 1e-3) if (np.isfinite(t_ms) and np.isfinite(t0_ms)) else (k * dt)
    if t_sec < ins_start_blend_sec and np.isfinite(v_gnss_n) and np.isfinite(v_gnss_e):
        blend = 1.0 - (t_sec / ins_start_blend_sec)
        v_n_ins[k] = (1.0 - blend) * v_n_ins[k] + blend * v_gnss_n
        v_e_ins[k] = (1.0 - blend) * v_e_ins[k] + blend * v_gnss_e

    # Интегрируем положение ИНС в локальных метрах.
    lat_ins[k] = lat_ins[k - 1] + v_n_ins[k] * dt_k
    lon_ins[k] = lon_ins[k - 1] + v_e_ins[k] * dt_k

    # Жесткая привязка позиции ИНС к GNSS по заданному интервалу.
    if np.isfinite(lat) and np.isfinite(lon) and (t_sec - last_ins_hard_sync_sec >= ins_hard_sync_interval_sec):
        lon_gnss_m, lat_gnss_m = geodetic_to_local_m(lat, lon, lat_ref_deg, lon_ref_deg)
        lat_ins[k] = lat_gnss_m
        lon_ins[k] = lon_gnss_m
        last_ins_hard_sync_sec = t_sec


    # Вычисляем остатки между предсказанной ИНС-позицией и GNSS-позицией.
    # lon_ins и lat_ins здесь хранят смещения в метрах относительно опорной точки.
    lon_pred_deg = lon_ref_deg + lon_meters_to_deg(lon_ins[k], lat_ref_deg=lat_ref_deg, height_m=height)
    lat_pred_deg = lat_ref_deg + lat_meters_to_deg(lat_ins[k], lat_ref_deg=lat_ref_deg, height_m=height)
    d_lon_m = lon_deg_to_meters(lon_pred_deg - lon, lat_ref_deg=lat_ref_deg, height_m=height)
    d_lat_m = lat_deg_to_meters(lat_pred_deg - lat, lat_ref_deg=lat_ref_deg, height_m=height)

    # Проверка на выбросы GNSS-позиции.
    if np.isfinite(d_lon_m) and abs(d_lon_m) > max_gnss_pos_residual_m:
        d_lon_m = np.nan
    if np.isfinite(d_lat_m) and abs(d_lat_m) > max_gnss_pos_residual_m:
        d_lat_m = np.nan

    # z_ve/z_vn — разность ИНС-скорости и GNSS-скорости.
    z_ve = (v_e_ins[k] - v_gnss_e) if np.isfinite(v_gnss_e) else np.nan
    z_vn = (v_n_ins[k] - v_gnss_n) if np.isfinite(v_gnss_n) else np.nan
    Z[k] = np.array([d_lon_m, d_lat_m, z_ve, z_vn])

    C = transf

    # Линейная аппроксимация динамики ошибки состояния. Все элементы
    # F[k] отражают, как текущие погрешности влияют на изменение состояния.
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
    Q[k] = build_Q_from_imu_noise(C, dt_k)
    
    


    # H выбирает из состояния те компоненты, которые напрямую измеряются:
    # [pos east, pos north, vel east, vel north]. Остальные состояния скрыты.
    H[k] = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])

    G[k] = build_G(C)
    Fi[k] = np.eye(13) + F[k] * dt_k  # дискретизация перехода состояния
    Ge[k] = G[k] * dt_k  # дискретизация влияния шума на состояние

    # Фильтр Калмана: предсказание и обновление состояния на основе измерений.
    x_pred = Fi[k] @ X[k - 1]
    S[k] = (Fi[k] @ P[k - 1] @ np.transpose(Fi[k])) + Q[k]

    h_rows = []
    z_vals = []
    r_vals = []

    # Формируем эффективную наблюдательную модель по доступным измерениям.
    # Если какая-либо входная компонента отсутствует, то она исключается из обновления.
    if np.isfinite(d_lon_m) and np.isfinite(d_lat_m):
        h_rows.append(H[k][0]); z_vals.append(d_lon_m); r_vals.append(sigma_pos_m**2)
        h_rows.append(H[k][1]); z_vals.append(d_lat_m); r_vals.append(sigma_pos_m**2)
    if np.isfinite(z_ve):
        h_rows.append(H[k][2]); z_vals.append(z_ve); r_vals.append(sigma_vel_mps**2)
    if np.isfinite(z_vn):
        h_rows.append(H[k][3]); z_vals.append(z_vn); r_vals.append(sigma_vel_mps**2)

    if h_rows:
        H_eff = np.vstack(h_rows)
        z_eff = np.array(z_vals)
        # Ковариация ошибки измерений.
        R_eff = np.diag(r_vals)
        innov_cov = H_eff @ S[k] @ H_eff.T + R_eff
        K_eff = S[k] @ H_eff.T @ np.linalg.inv(innov_cov)
        X[k] = x_pred + K_eff @ (z_eff - H_eff @ x_pred)
        P[k] = (np.eye(13) - K_eff @ H_eff) @ S[k]
    else:
        X[k] = x_pred
        P[k] = S[k]



 #-----------------Код который ЧИСТО интегрирует-------------------------
    # Здесь считается чистая ИНС-траектория без учета фильтра состояния X.
    lat_real_ins[k] = lat_real_ins[k-1]
    lon_real_ins[k] = lon_real_ins[k-1]
    v_n_pure[k] = v_n_pure[k-1]
    v_e_pure[k] = v_e_pure[k-1]

    # Получаем данные
    ax = inputs[k]["gFx"]
    ay = inputs[k]["gFy"]
    az = inputs[k]["gFz"]
    roll = np.deg2rad(inputs[k]["roll"])
    pitch = np.deg2rad(inputs[k]["pitch"])
    yaw = np.deg2rad(inputs[k]["yaw"])

    # Пропускаем если нет валидных данных
    if not np.all(np.isfinite([ax, ay, az, roll, pitch, yaw])):
        continue

    # Расчет dt
    dt_k = sample_dt_s(inputs[k], inputs[k-1], dt)

    # Матрица поворота связанная -> нормальная (NEH)
    C = body_to_normal_matrix(yaw, pitch, roll)

    # Ускорение в связанной СК
    a_body = np.array([ax, ay, az])

    # Гравитация в связанной СК
    gravity_body = C.T @ gravity_nav

    # Линейное ускорение в нормальной СК [North, East, Down]
    a_nav = C @ (a_body - gravity_body)

    a_north = a_nav[0]
    a_east = a_nav[1]

    # Интегрируем скорости
    v_n_pure[k] = v_n_pure[k-1] + a_north * dt_k
    v_e_pure[k] = v_e_pure[k-1] + a_east * dt_k

    # Интегрируем позицию в метрах
    north_m += v_n_pure[k] * dt_k
    east_m += v_e_pure[k] * dt_k

    # Метры -> градусы
    delta_lat_deg = lat_meters_to_deg(north_m, lat_ref_deg, height)
    delta_lon_deg = lon_meters_to_deg(east_m, lat_ref_deg, height)

    lat_real_ins[k] = lat_ref_deg + delta_lat_deg
    lon_real_ins[k] = lon_ref_deg + delta_lon_deg
 #------------------------------------------


    # Преобразуем состояние ошибки обратно в геодезические координаты для отображения.
    lon_kalm_m = lon_ins[k] - X[k][0]
    lat_kalm_m = lat_ins[k] - X[k][1]
    lat_kalm[k], lon_kalm[k] = local_m_to_geodetic(lon_kalm_m, lat_kalm_m, lat_ref_deg, lon_ref_deg)

    # Скорости после фильтрации по смещению ошибок.
    v_e_kalm[k] = v_e_ins[k] - X[k][2]
    v_n_kalm[k] = v_n_ins[k] - X[k][3]

    # Корректируем оценки ИНС на найденные ошибки состояния X.
    v_e_ins[k] = v_e_ins[k] - X[k][2]
    v_n_ins[k] = v_n_ins[k] - X[k][3]
    lon_ins[k] = lon_ins[k] - X[k][0]
    lat_ins[k] = lat_ins[k] - X[k][1]








# --------------------- вывод результатов на карту --------------------
import folium
import webbrowser


def make_points(lat_array, lon_array):
    """Convert arrays of latitude/longitude into folium-friendly point lists."""
    points = []
    for lat, lon in zip(lat_array, lon_array):
        if np.isfinite(lat) and np.isfinite(lon):
            if abs(lat) > 0.000001 and abs(lon) > 0.000001:
                points.append([lat, lon])
    return points


# GNSS-координаты из CSV

gps_lat = np.array([row["lat"] for row in inputs])
gps_lon = np.array([row["lon"] for row in inputs])






# Переводим локальные метры ИНС обратно в геодезические координаты для карты.
for k in range(len(lat_ins)):
    lat_ins[k], lon_ins[k] = local_m_to_geodetic(lon_ins[k], lat_ins[k], lat_ref_deg, lon_ref_deg)

gps_points = make_points(gps_lat, gps_lon)
ins_points = make_points(lat_real_ins, lon_real_ins)
kalman_points = make_points(lat_kalm, lon_kalm)

# ins_points = np.zeros((len(gps_lat), 2))
# kalman_points = np.zeros((len(gps_lat), 2))
# gps_points = np.zeros((len(gps_lat), 2))

# Центр карты — первая нормальная GNSS-точка
if not gps_points:
    raise ValueError("No valid GNSS points for map rendering.")

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