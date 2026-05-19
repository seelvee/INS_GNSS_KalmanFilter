import csv
from dataclasses import dataclass
from pathlib import Path
import webbrowser

import folium
import numpy as np


# --------------------------------- Config ---------------------------------


@dataclass
class FilterConfig:
    earth_e2: float = 6.69437999014e-3
    earth_a: float = 6378137.0
    height_m: float = 167.0
    omega_shuler: float = 0.0012407  # rad/s
    gravity_nav: tuple[float, float, float] = (0.0, 0.0, -9.80665)
    sigma_pos_m: float = 3.0
    sigma_vel_mps: float = 3.0
    # Process noise from IMU calibration.
    accel_noise_std_mps2: tuple[float, float, float] = (
        0.026603212389077466,
        0.018265213187570104,
        0.028088268862857684,
    )
    gyro_noise_std_rad_s: tuple[float, float, float] = (
        0.0179383401865371,
        0.08979950425318393,
        0.0005319062935916511,
    )


CFG = FilterConfig()


# --------------------------------- IO ---------------------------------


def _to_float(token: str) -> float:
    value = token.strip().lower()
    if value in {"", "nan", "none"}:
        return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan


def parse_csv(csv_path: str | Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    header: list[str] | None = None

    with Path(csv_path).open("r", encoding="utf-8", newline="") as file:
        for raw_row in csv.reader(file):
            if not raw_row:
                continue
            if raw_row[0].strip().startswith("#"):
                continue

            if header is None:
                header = [c.strip() for c in raw_row]
                continue

            row: dict[str, float] = {}
            for i, col in enumerate(header):
                if i < len(raw_row):
                    row[col] = _to_float(raw_row[i])
                elif col in {"roll_deg", "pitch_deg", "yaw_deg"}:
                    row[col] = 0.0
                else:
                    row[col] = np.nan
            rows.append(row)

    return rows


# --------------------------------- Geodesy ---------------------------------


def radii_of_curvature(lat_deg: float, height_m: float) -> tuple[float, float]:
    lat_rad = np.deg2rad(lat_deg)
    sin_lat = np.sin(lat_rad)
    rho_n = CFG.earth_a * (1 - CFG.earth_e2) / (1 - CFG.earth_e2 * sin_lat**2) ** 1.5 + height_m
    rho_e = CFG.earth_a / np.sqrt(1 - CFG.earth_e2 * sin_lat**2) + height_m
    return rho_n, rho_e


def geodetic_to_local_m(
    lat_deg: float,
    lon_deg: float,
    lat0_deg: float,
    lon0_deg: float,
    height_m: float,
) -> tuple[float, float]:
    rho_n, rho_e = radii_of_curvature(lat0_deg, height_m)
    d_lat = np.deg2rad(lat_deg - lat0_deg)
    d_lon = np.deg2rad(lon_deg - lon0_deg)
    north_m = d_lat * rho_n
    east_m = d_lon * rho_e * np.cos(np.deg2rad(lat0_deg))
    return east_m, north_m


def local_m_to_geodetic(
    east_m: float,
    north_m: float,
    lat0_deg: float,
    lon0_deg: float,
    height_m: float,
) -> tuple[float, float]:
    rho_n, rho_e = radii_of_curvature(lat0_deg, height_m)
    lat = lat0_deg + np.rad2deg(north_m / rho_n)
    lon = lon0_deg + np.rad2deg(east_m / (rho_e * np.cos(np.deg2rad(lat0_deg))))
    return lat, lon


# --------------------------------- INS/KF math ---------------------------------


def body_to_normal_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    c_psi, s_psi = np.cos(yaw), np.sin(yaw)
    c_theta, s_theta = np.cos(pitch), np.sin(pitch)
    c_gamma, s_gamma = np.cos(roll), np.sin(roll)

    return np.array(
        [
            [
                c_theta * c_psi,
                s_gamma * s_theta * c_psi - c_gamma * s_psi,
                c_gamma * s_theta * c_psi + s_gamma * s_psi,
            ],
            [
                c_theta * s_psi,
                s_gamma * s_theta * s_psi + c_gamma * c_psi,
                c_gamma * s_theta * s_psi - s_gamma * c_psi,
            ],
            [-s_theta, s_gamma * c_theta, c_gamma * c_theta],
        ],
        dtype=float,
    )


def build_g(C: np.ndarray) -> np.ndarray:
    G = np.zeros((13, 6), dtype=float)
    G[2:4, 3:6] = C[0:2, :]
    G[4:7, 0:3] = C
    return G


def build_q(C: np.ndarray, dt: float) -> np.ndarray:
    accel = np.array(CFG.accel_noise_std_mps2, dtype=float)
    gyro = np.array(CFG.gyro_noise_std_rad_s, dtype=float)
    qw = np.diag(
        [
            gyro[0] ** 2,
            gyro[1] ** 2,
            gyro[2] ** 2,
            accel[0] ** 2,
            accel[1] ** 2,
            accel[2] ** 2,
        ]
    )
    G = build_g(C)
    return G @ qw @ G.T * dt**2


def first_valid_gnss(rows: list[dict[str, float]]) -> tuple[float, float]:
    for row in rows:
        lat, lon = row.get("lat", np.nan), row.get("lon", np.nan)
        if np.isfinite(lat) and np.isfinite(lon):
            return lat, lon
    raise ValueError("No valid GNSS points in CSV.")


# --------------------------------- Main ---------------------------------


def run_filter(csv_path: str | Path) -> dict[str, np.ndarray]:
    rows = parse_csv(csv_path)
    if len(rows) < 2:
        raise ValueError("Not enough rows in CSV.")

    lat0, lon0 = first_valid_gnss(rows)
    N = len(rows)
    n = 13
    m = 4  # GNSS position + velocity updates (East/North, meters and m/s)

    gravity_nav = np.array(CFG.gravity_nav, dtype=float)

    # INS state in local EN (meters).
    east_ins = np.zeros(N, dtype=float)
    north_ins = np.zeros(N, dtype=float)
    v_e_ins = np.zeros(N, dtype=float)
    v_n_ins = np.zeros(N, dtype=float)

    # KF arrays.
    F = np.zeros((N, n, n), dtype=float)
    Phi = np.zeros((N, n, n), dtype=float)
    Q = np.zeros((N, n, n), dtype=float)
    P = np.zeros((N, n, n), dtype=float)
    H = np.zeros((N, m, n), dtype=float)
    K = np.zeros((N, n, m), dtype=float)
    S = np.zeros((N, n, n), dtype=float)
    X = np.zeros((N, n), dtype=float)
    Z = np.zeros((N, m), dtype=float)
    v_e_gps = np.full(N, np.nan, dtype=float)
    v_n_gps = np.full(N, np.nan, dtype=float)

    # Filter init.
    P[0] = np.eye(n, dtype=float) * 0.04
    Phi[0] = np.eye(n, dtype=float)

    # Output tracks in geodetic coordinates.
    gps_lat = np.array([r.get("lat", np.nan) for r in rows], dtype=float)
    gps_lon = np.array([r.get("lon", np.nan) for r in rows], dtype=float)
    ins_lat = np.full(N, np.nan, dtype=float)
    ins_lon = np.full(N, np.nan, dtype=float)
    kal_lat = np.full(N, np.nan, dtype=float)
    kal_lon = np.full(N, np.nan, dtype=float)

    # Precompute GNSS velocity in local frame from finite differences.
    east_gps_track = np.full(N, np.nan, dtype=float)
    north_gps_track = np.full(N, np.nan, dtype=float)
    for i in range(N):
        lat_i = rows[i].get("lat", np.nan)
        lon_i = rows[i].get("lon", np.nan)
        if np.isfinite(lat_i) and np.isfinite(lon_i):
            east_gps_track[i], north_gps_track[i] = geodetic_to_local_m(lat_i, lon_i, lat0, lon0, CFG.height_m)
    for i in range(1, N):
        t_i = rows[i].get("time_ms", np.nan)
        t_prev = rows[i - 1].get("time_ms", np.nan)
        if np.isfinite(t_i) and np.isfinite(t_prev) and np.isfinite(east_gps_track[i]) and np.isfinite(east_gps_track[i - 1]):
            dt_i = max(1e-3, (t_i - t_prev) / 1000.0)
            v_e_gps[i] = (east_gps_track[i] - east_gps_track[i - 1]) / dt_i
            v_n_gps[i] = (north_gps_track[i] - north_gps_track[i - 1]) / dt_i

    for k in range(1, N):
        t_now = rows[k].get("time_ms", np.nan)
        t_prev = rows[k - 1].get("time_ms", np.nan)
        if np.isfinite(t_now) and np.isfinite(t_prev):
            dt = max(1e-3, (t_now - t_prev) / 1000.0)
        else:
            dt = 0.01

        lat = rows[k].get("lat", np.nan)
        lon = rows[k].get("lon", np.nan)
        ax = rows[k].get("ax", np.nan)
        ay = rows[k].get("ay", np.nan)
        az = rows[k].get("az", np.nan)
        gx = rows[k].get("gx", np.nan)
        gy = rows[k].get("gy", np.nan)
        gz = rows[k].get("gz", np.nan)
        roll_deg = rows[k].get("roll_deg", 0.0)
        pitch_deg = rows[k].get("pitch_deg", 0.0)
        yaw_deg = rows[k].get("yaw_deg", 0.0)

        # Skip propagation if IMU data is absent.
        if not np.all(np.isfinite([ax, ay, az])):
            east_ins[k] = east_ins[k - 1]
            north_ins[k] = north_ins[k - 1]
            v_e_ins[k] = v_e_ins[k - 1]
            v_n_ins[k] = v_n_ins[k - 1]
            continue

        # Angles to radians, gyro deg/s -> rad/s.
        roll = np.deg2rad(0.0 if not np.isfinite(roll_deg) else roll_deg)
        pitch = np.deg2rad(0.0 if not np.isfinite(pitch_deg) else pitch_deg)
        yaw = np.deg2rad(0.0 if not np.isfinite(yaw_deg) else yaw_deg)
        gx = np.deg2rad(0.0 if not np.isfinite(gx) else gx)
        gy = np.deg2rad(0.0 if not np.isfinite(gy) else gy)
        gz = np.deg2rad(0.0 if not np.isfinite(gz) else gz)

        C = body_to_normal_matrix(yaw, pitch, roll)

        # Bias-compensated acceleration.
        a_body = np.array([ax - X[k - 1, 10], ay - X[k - 1, 11], az - X[k - 1, 12]], dtype=float)
        a_nav = C @ a_body + gravity_nav
        omega_nav = C @ np.array([gx, gy, gz], dtype=float)

        n_e, n_n = float(np.clip(a_nav[1], -5.0, 5.0)), float(np.clip(a_nav[0], -5.0, 5.0))

        v_n_ins[k] = v_n_ins[k - 1] + n_n * dt
        v_e_ins[k] = v_e_ins[k - 1] + n_e * dt
        north_ins[k] = north_ins[k - 1] + v_n_ins[k] * dt
        east_ins[k] = east_ins[k - 1] + v_e_ins[k] * dt

        # Build dynamics at current latitude if possible.
        lat_for_radius = lat if np.isfinite(lat) else lat0
        rho_n, rho_e = radii_of_curvature(lat_for_radius, CFG.height_m)

        # Stable linearized model: keep kinematics and accel-bias coupling.
        # Full nonlinear coupling terms are intentionally omitted here to
        # prevent numerical blow-up on noisy low-cost IMU datasets.
        F[k] = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C[0, 0], C[0, 1], C[0, 2]],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, C[1, 0], C[1, 1], C[1, 2]],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=float,
        )

        Phi[k] = np.eye(n, dtype=float) + F[k] * dt
        Q[k] = build_q(C, dt)

        # Prediction.
        X_pred = Phi[k - 1] @ X[k - 1]
        S[k] = Phi[k - 1] @ P[k - 1] @ Phi[k - 1].T + Q[k - 1]

        # Position/velocity measurement when GNSS is valid.
        if np.isfinite(lat) and np.isfinite(lon):
            east_gps, north_gps = geodetic_to_local_m(lat, lon, lat0, lon0, CFG.height_m)
            vel_e_meas = 0.0 if not np.isfinite(v_e_gps[k]) else v_e_ins[k] - v_e_gps[k]
            vel_n_meas = 0.0 if not np.isfinite(v_n_gps[k]) else v_n_ins[k] - v_n_gps[k]
            Z[k] = np.array([east_ins[k] - east_gps, north_ins[k] - north_gps, vel_e_meas, vel_n_meas], dtype=float)

            H[k, 0, 0] = 1.0
            H[k, 1, 1] = 1.0
            H[k, 2, 2] = 1.0
            H[k, 3, 3] = 1.0

            R = np.diag(
                [
                    CFG.sigma_pos_m**2,
                    CFG.sigma_pos_m**2,
                    CFG.sigma_vel_mps**2,
                    CFG.sigma_vel_mps**2,
                ]
            ).astype(float)
            innovation_cov = H[k] @ S[k] @ H[k].T + R
            K[k] = S[k] @ H[k].T @ np.linalg.pinv(innovation_cov)
            innovation = Z[k] - H[k] @ X_pred
            X[k] = X_pred + K[k] @ innovation

            I = np.eye(n, dtype=float)
            # Joseph form for numerical stability.
            P[k] = (I - K[k] @ H[k]) @ S[k] @ (I - K[k] @ H[k]).T + K[k] @ R @ K[k].T
        else:
            X[k] = X_pred
            P[k] = S[k]

        # Hard safety against divergence from bad packets/outliers.
        if not np.all(np.isfinite(X[k])) or np.max(np.abs(X[k])) > 1e6:
            X[k] = X[k - 1]
        if not np.all(np.isfinite(P[k])) or np.max(np.abs(P[k])) > 1e12:
            P[k] = P[k - 1]

        # Closed-loop correction from estimated navigation errors.
        v_e_ins[k] -= float(np.clip(X[k, 2], -1.0, 1.0))
        v_n_ins[k] -= float(np.clip(X[k, 3], -1.0, 1.0))
        east_ins[k] -= float(np.clip(X[k, 0], -3.0, 3.0))
        north_ins[k] -= float(np.clip(X[k, 1], -3.0, 3.0))

        # Save tracks in degrees.
        ins_lat[k], ins_lon[k] = local_m_to_geodetic(east_ins[k], north_ins[k], lat0, lon0, CFG.height_m)
        kal_lat[k], kal_lon[k] = local_m_to_geodetic(
            east_ins[k] - X[k, 0],
            north_ins[k] - X[k, 1],
            lat0,
            lon0,
            CFG.height_m,
        )

    return {
        "gps_lat": gps_lat,
        "gps_lon": gps_lon,
        "ins_lat": ins_lat,
        "ins_lon": ins_lon,
        "kal_lat": kal_lat,
        "kal_lon": kal_lon,
    }


# --------------------------------- Visualization ---------------------------------


def make_points(lat_array: np.ndarray, lon_array: np.ndarray) -> list[list[float]]:
    points: list[list[float]] = []
    for lat, lon in zip(lat_array, lon_array):
        if np.isfinite(lat) and np.isfinite(lon):
            if abs(lat) > 1e-6 and abs(lon) > 1e-6:
                points.append([float(lat), float(lon)])
    return points


def save_map(result: dict[str, np.ndarray], output_path: Path) -> None:
    gps_points = make_points(result["gps_lat"], result["gps_lon"])
    ins_points = make_points(result["ins_lat"], result["ins_lon"])
    kalman_points = make_points(result["kal_lat"], result["kal_lon"])

    if not gps_points:
        raise ValueError("No valid GPS points to render map.")

    m = folium.Map(location=gps_points[0], zoom_start=18, tiles="OpenStreetMap")

    folium.PolyLine(gps_points, tooltip="GNSS", color="blue", weight=5).add_to(m)
    if len(ins_points) > 1:
        folium.PolyLine(ins_points, tooltip="INS", color="red", weight=3).add_to(m)
    if len(kalman_points) > 1:
        folium.PolyLine(kalman_points, tooltip="Kalman", color="green", weight=4).add_to(m)

    folium.Marker(gps_points[0], tooltip="Start", popup="Начало").add_to(m)
    folium.Marker(gps_points[-1], tooltip="Finish", popup="Конец").add_to(m)

    m.save(output_path)


if __name__ == "__main__":
    source = Path("arduino_ins_gnss/ins_gnss_20260506_211949.csv")
    out_map = Path("trajectory_map_review.html")

    result = run_filter(source)
    save_map(result, out_map)
    webbrowser.open(out_map.resolve().as_uri())
