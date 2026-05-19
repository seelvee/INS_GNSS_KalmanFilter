import csv
from pathlib import Path
import numpy as np

omega_shu =  0.0012407 # rad/s
height = 167 # высота в метрах
e2 = 6.69437999014e-3
sigma_pos_m = 1.10
sigma_vel_mps = 0.110
lat_msk = 55.36


gravity_nav = np.array([0.0, 0.0, -9.80665])
dt = 0.01
max_gnss_pos_residual_m = 1500.0
min_dt_s = 0.005
max_dt_s = 0.2
ins_start_blend_sec = 15.0
ins_static_speed_threshold_mps = 0.6
ins_bias_calib_max_samples = 250
ins_accel_lpf_alpha = 0.15
ins_hard_sync_interval_sec = 1000000.0



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


def sample_dt_s(curr_row: dict[str, float], prev_row: dict[str, float], default_dt_s: float = dt) -> float:
    """Get sample period from time_ms if available, otherwise use default."""
    t_curr = curr_row.get("time_ms", np.nan)
    t_prev = prev_row.get("time_ms", np.nan)
    if np.isfinite(t_curr) and np.isfinite(t_prev):
        dt_s = (t_curr - t_prev) * 1e-3
        if min_dt_s <= dt_s <= max_dt_s:
            return float(dt_s)
    return default_dt_s


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
        0.126603212389077466,
        0.118265213187570104,
        0.128088268862857684
    ])

    gyro_noise_std_rad_s = np.array([
        0.179383401865371,
        0.08979950425318393,
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

    Q = G @ Qw @ G.T * dt

    return Q




# --------------------------------- Начало программы -------------------------------

path = "arduino_ins_gnss\\multi_1778262478.318379.csv"
inputs = parse_csv(path)

if not inputs:
    raise ValueError("CSV is empty or has no data rows.")

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


P[0] = np.diag([
    0.10**2,     # lon/east error, m^2
    01.10**2,     # lat/north error, m^2
    1.0**2,      # v_e error, (m/s)^2
    1.0**2,      # v_n error
    np.deg2rad(5.0)**2,   # alpha
    np.deg2rad(5.0)**2,   # beta
    np.deg2rad(10.0)**2,  # gamma / yaw
    0.01**2,    # gyro bias x
    0.01**2,    # gyro bias y
    0.01**2,    # gyro bias z
    0.2**2,     # accel bias x
    0.2**2,     # accel bias y
    0.2**2      # accel bias z
])
Fi[0] = np.eye(n)


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
    # По умолчанию держим предыдущее состояние, если текущая строка данных неполная.
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

    # lat = inputs[k]["lat"]
    # lon = inputs[k]["lon"]
    # dt_k = sample_dt_s(inputs[k], inputs[k - 1], dt)
    # a_x = inputs[k]["ax"]
    # a_y = inputs[k]["ay"]
    # a_z = inputs[k]["az"]
    # g_x = np.deg2rad(inputs[k]["gx"])
    # g_y = np.deg2rad(inputs[k]["gy"])
    # g_z = np.deg2rad(inputs[k]["gz"])
    # v_gnss_e = inputs[k]["gnss_ve_mps"]
    # v_gnss_n = inputs[k]["gnss_vn_mps"]
    # gnss_speed = inputs[k].get("gnss_speed_mps", np.nan)

    lat = inputs[k]["lat"]
    lon = inputs[k]["lon"]
    dt_k = sample_dt_s(inputs[k], inputs[k - 1], dt)
    a_x = inputs[k]["gFx"]
    a_y = inputs[k]["gFy"]
    a_z = inputs[k]["gFz"]
    g_x = np.deg2rad(inputs[k]["wx"])
    g_y = np.deg2rad(inputs[k]["wy"])
    g_z = np.deg2rad(inputs[k]["wz"])

    roll = np.deg2rad(inputs[k]["roll"])
    pitch = np.deg2rad(inputs[k]["pitch"])
    yaw = np.deg2rad(inputs[k]["yaw"])
    gnss_speed = inputs[k]["speed"]
    v_gnss_e = gnss_speed * np.sin(yaw)
    v_gnss_n = gnss_speed * np.cos(yaw)
    
    
    


    
    
    if not np.all(np.isfinite([lat, lon, a_x, a_y, a_z, g_x, g_y, g_z, roll, pitch, yaw])):
        continue

    # Сырые ускорения с учетом оцененных bias акселерометра.
    # В логе Arduino передается dmpGetAccel(), т.е. ускорение включает гравитацию.
    a = np.array([a_x - X[k - 1][10], a_y - X[k - 1][11], a_z - X[k - 1][12]])
    omega = np.array([g_x, g_y, g_z])
    transf = body_to_normal_matrix(yaw, pitch, roll)

    # Убираем g в связанной СК, затем переводим линейное ускорение в навигационную СК.
    # Это предотвращает "двойной учет" гравитации и взрывной дрейф скорости.
    gravity_body = transf.T @ gravity_nav
    a_lin_body = a - gravity_body
    a = transf @ a_lin_body
    a_nav_lpf_prev = (1.0 - ins_accel_lpf_alpha) * a_nav_lpf_prev + ins_accel_lpf_alpha * a
    a = a_nav_lpf_prev.copy()

    # На старте оцениваем остаточный bias ускорения в навигационной СК на квазистатике.
    if (
        accel_bias_count < ins_bias_calib_max_samples
        and np.isfinite(gnss_speed)
        and gnss_speed <= ins_static_speed_threshold_mps
    ):
        accel_bias_nav = (accel_bias_nav * accel_bias_count + a) / (accel_bias_count + 1)
        accel_bias_count += 1

    a = a - accel_bias_nav
    omega = transf @ omega
    
    n_e = a[1]
    n_H = a[2]
    n_N = a[0]

    # --------------- Часть кода которая интегрирует ускорение -----------------------
    v_n_ins[k] = v_n_ins[k-1] + n_N * dt_k # м/с
    v_e_ins[k] = v_e_ins[k-1] + n_e * dt_k # м/с

    # Мягкая привязка скорости ИНС к GNSS в начале траектории, чтобы убрать стартовые колебания.
    t_ms = inputs[k].get("time_ms", np.nan)
    t0_ms = inputs[0].get("time_ms", np.nan)
    t_sec = ((t_ms - t0_ms) * 1e-3) if (np.isfinite(t_ms) and np.isfinite(t0_ms)) else (k * dt)
    if t_sec < ins_start_blend_sec and np.isfinite(v_gnss_n) and np.isfinite(v_gnss_e):
        blend = 1.0 - (t_sec / ins_start_blend_sec)
        v_n_ins[k] = (1.0 - blend) * v_n_ins[k] + blend * v_gnss_n
        v_e_ins[k] = (1.0 - blend) * v_e_ins[k] + blend * v_gnss_e

    lat_ins[k] = lat_ins[k - 1] + (v_n_ins[k]) * dt_k # м
    lon_ins[k] = lon_ins[k - 1] + (v_e_ins[k]) * dt_k # м

    # Жесткая связь: периодически принудительно притягиваем ИНС-координаты к GNSS.
    if np.isfinite(lat) and np.isfinite(lon) and (t_sec - last_ins_hard_sync_sec >= ins_hard_sync_interval_sec):
        lon_gnss_m, lat_gnss_m = geodetic_to_local_m(lat, lon, lat_ref_deg, lon_ref_deg)
        lat_ins[k] = lat_gnss_m
        lon_ins[k] = lon_gnss_m
        last_ins_hard_sync_sec = t_sec
    # --------------------------------------------------------------------------------


    # ВОТ ТУТ ПЕРЕСМОТРИ ВСЕ ЕЩЕ РАЗ !!!!!!!!!!!!!!!

    lon_pred_deg = lon_ref_deg + lon_meters_to_deg(lon_ins[k], lat_ref_deg=lat_ref_deg, height_m=height)
    lat_pred_deg = lat_ref_deg + lat_meters_to_deg(lat_ins[k], lat_ref_deg=lat_ref_deg, height_m=height)
    d_lon_m = lon_deg_to_meters(lon_pred_deg - lon, lat_ref_deg=lat_ref_deg, height_m=height)
    d_lat_m = lat_deg_to_meters(lat_pred_deg - lat, lat_ref_deg=lat_ref_deg, height_m=height)

    if np.isfinite(d_lon_m) and abs(d_lon_m) > max_gnss_pos_residual_m:
        d_lon_m = np.nan
    if np.isfinite(d_lat_m) and abs(d_lat_m) > max_gnss_pos_residual_m:
        d_lat_m = np.nan

    # Z[k] = np.array([
    #     d_lon_m,              # ошибка по East, м
    #     d_lat_m,              # ошибка по North, м
    #     0.000, # ошибка восточной скорости
    #     0.000  # ошибка северной скорости
    # ])

    z_ve = (v_e_ins[k] - v_gnss_e) if np.isfinite(v_gnss_e) else np.nan
    z_vn = (v_n_ins[k] - v_gnss_n) if np.isfinite(v_gnss_n) else np.nan
    Z[k] = np.array([d_lon_m, d_lat_m, z_ve, z_vn])


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
    Q[k] = build_Q_from_imu_noise(C, dt_k)
    
    
    H[k] = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # H[k] = np.array([
    # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    # ])

    G[k] = build_G(C)
    Fi[k] = np.eye(13) + F[k] * dt_k
    Ge[k] = G[k] * dt_k

    # фильтр калмана ----------------
    x_pred = Fi[k] @ X[k - 1]
    S[k] = (Fi[k] @ P[k - 1] @ np.transpose(Fi[k])) + Q[k]

    h_rows = []
    z_vals = []
    r_vals = []
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
        R_eff = np.diag(r_vals)
        innov_cov = H_eff @ S[k] @ H_eff.T + R_eff
        K_eff = S[k] @ H_eff.T @ np.linalg.inv(innov_cov)
        X[k] = x_pred + K_eff @ (z_eff - H_eff @ x_pred)
        P[k] = (np.eye(13) - K_eff @ H_eff) @ S[k]
    else:
        X[k] = x_pred
        P[k] = S[k]

    lon_kalm_m = lon_ins[k] - X[k][0]
    lat_kalm_m = lat_ins[k] - X[k][1]
    lat_kalm[k], lon_kalm[k] = local_m_to_geodetic(lon_kalm_m, lat_kalm_m, lat_ref_deg, lon_ref_deg)

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






for k in range(len(lat_ins)):
    lat_ins[k], lon_ins[k] = local_m_to_geodetic(lon_ins[k], lat_ins[k], lat_ref_deg, lon_ref_deg)

gps_points = make_points(gps_lat, gps_lon)
ins_points = make_points(lat_ins, lon_ins)
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