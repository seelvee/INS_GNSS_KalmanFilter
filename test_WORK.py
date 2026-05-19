"""
Парсинг CSV с телеметрией INS/GNSS, обработка алгоритмами из main/filterKalman.py (без импорта
модуля целиком — у него исполняется свой блок «Начало программы»), вывод треков на карту Folium.
"""

from __future__ import annotations

import sys
from pathlib import Path

import folium
import numpy as np

ROOT = Path(__file__).resolve().parent
FK_PATH = ROOT / "main" / "filterKalman.py"
CSV_PATH = ROOT / "arduino_ins_gnss" / "ins_gnss_20260430_200111.csv"
OUT_HTML = ROOT / f"{CSV_PATH.stem}_map.html"


def _load_filter_kalman_api():
    """Поднимает функции и константы из filterKalman.py без выполнения основного скрипта."""
    text = FK_PATH.read_text(encoding="utf-8")
    marker = "# --------------------------------- Начало программы -------------------------------"
    if marker not in text:
        raise RuntimeError(f"Маркер блока не найден в {FK_PATH}")
    head = text.split(marker, 1)[0]
    ns: dict = {"__builtins__": __builtins__, "np": np}
    exec(compile(head + "\n", str(FK_PATH), "exec"), ns)
    return ns


def _forward_fill_coords(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    lat = np.array([r["lat"] for r in rows], dtype=float)
    lon = np.array([r["lon"] for r in rows], dtype=float)
    i0 = None
    for i in range(len(lat)):
        if np.isfinite(lat[i]) and np.isfinite(lon[i]):
            i0 = i
            break
    if i0 is None:
        raise ValueError("В CSV нет ни одной строки с валидными lat/lon")
    fill_lat, fill_lon = lat[i0], lon[i0]
    for i in range(i0):
        lat[i], lon[i] = fill_lat, fill_lon
    for i in range(i0 + 1, len(lat)):
        if not np.isfinite(lat[i]) or not np.isfinite(lon[i]):
            lat[i], lon[i] = lat[i - 1], lon[i - 1]
    return lat, lon


_IMU_ORIENTATION_KEYS = ("ax", "ay", "az", "gx", "gy", "gz", "roll_deg", "pitch_deg", "yaw_deg")


def _forward_fill_imu(rows: list[dict[str, float]]) -> None:
    """Заполняет пропуски в IMU/углах (в логе часто первая строка — nan до первого пакета)."""
    carry: dict[str, float] = {k: float("nan") for k in _IMU_ORIENTATION_KEYS}
    for r in rows:
        for k in _IMU_ORIENTATION_KEYS:
            v = float(r.get(k, float("nan")))
            if np.isfinite(v):
                carry[k] = v
            elif np.isfinite(carry[k]):
                r[k] = carry[k]
    carry = {k: float("nan") for k in _IMU_ORIENTATION_KEYS}
    for r in reversed(rows):
        for k in _IMU_ORIENTATION_KEYS:
            v = float(r.get(k, float("nan")))
            if np.isfinite(v):
                carry[k] = v
            elif np.isfinite(carry[k]):
                r[k] = carry[k]


def _finite_latlon(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
    return [(la, lo) for la, lo in coords if np.isfinite(la) and np.isfinite(lo)]


def run_kalman_track(fk: dict, inputs: list[dict[str, float]], lat_meas: np.ndarray, lon_meas: np.ndarray):
    omega_shu = fk["omega_shu"]
    height = fk["height"]
    e2 = fk["e2"]
    V_h = fk["V_h"]
    sigma_pos_m = fk["sigma_pos_m"]
    sigma_vel_mps = fk["sigma_vel_mps"]
    gravity_nav = fk["gravity_nav"]
    dt = fk["dt"]
    body_to_normal_matrix = fk["body_to_normal_matrix"]
    build_G = fk["build_G"]
    build_Q_from_imu_noise = fk["build_Q_from_imu_noise"]

    N = len(inputs)
    n, m = 13, 4

    v_n_ins = np.zeros(N)
    v_e_ins = np.zeros(N)
    F = np.zeros((N, n, n))
    Fi = np.zeros((N, n, n))
    Q = np.zeros((N, n, n))
    P = np.zeros((N, n, n))
    G = np.zeros((N, n, 6))
    Ge = np.zeros((N, n, 6))
    H = np.zeros((N, m, n))
    S = np.zeros((N, n, n))
    K = np.zeros((N, n, m))
    X = np.zeros((N, n))
    Z = np.zeros((N, 4))

    P[0] = np.eye(n)
    Fi[0] = np.eye(n)

    lat_ins = np.zeros(N)
    lon_ins = np.zeros(N)
    v_n_pure = np.zeros(N)
    v_e_pure = np.zeros(N)
    lat_ins_pure = np.zeros(N)
    lon_ins_pure = np.zeros(N)
    lat_kalm = np.full(N, np.nan)
    lon_kalm = np.full(N, np.nan)

    for k in range(N):
        lat = lat_meas[k]
        lon = lon_meas[k]
        row = inputs[k]
        a_x = row["ax"]
        a_y = row["ay"]
        a_z = row["az"]
        g_x = row["gx"]
        g_y = row["gy"]
        g_z = row["gz"]

        roll = np.deg2rad(row["roll_deg"])
        pitch = np.deg2rad(row["pitch_deg"])
        yaw = np.deg2rad(row["yaw_deg"])

        sin_phi = np.sin(np.deg2rad(lat))
        rho1 = 6378137.0 * (1 - e2) / (1 - e2 * sin_phi**2) ** (3 / 2) + height
        rho2 = 6378137.0 / np.sqrt(1 - e2 * sin_phi**2) + height

        transf = body_to_normal_matrix(yaw, pitch, roll)
        a_biased = np.array([a_x - X[k][10], a_y - X[k][11], a_z - X[k][12]])
        a = transf @ a_biased + gravity_nav
        a_pure = transf @ np.array([a_x, a_y, a_z]) + gravity_nav
        omega = np.array([g_x, g_y, g_z])
        omega = transf @ omega

        n_e = a[1]
        n_H = a[2]
        n_N = a[0]

        n_e_p = a_pure[1]
        n_N_p = a_pure[0]

        lat0 = lat_meas[k - 1] if k > 0 else lat_meas[0]
        lon0 = lon_meas[k - 1] if k > 0 else lon_meas[0]

        v_n_ins[k] = v_n_ins[k - 1] + n_N * dt
        v_e_ins[k] = v_e_ins[k - 1] + n_e * dt
        lat_ins[k] = lat0 + np.rad2deg((v_n_ins[k] / rho1) * dt)
        lon_ins[k] = lon0 + np.rad2deg((v_e_ins[k] / (rho2 * np.cos(np.deg2rad(lat_ins[k])))) * dt)

        lat0_p = lat_ins_pure[k - 1] if k > 0 else lat_meas[0]
        lon0_p = lon_ins_pure[k - 1] if k > 0 else lon_meas[0]
        v_n_pure[k] = v_n_pure[k - 1] + n_N_p * dt
        v_e_pure[k] = v_e_pure[k - 1] + n_e_p * dt
        lat_ins_pure[k] = lat0_p + np.rad2deg((v_n_pure[k] / rho1) * dt)
        lon_ins_pure[k] = lon0_p + np.rad2deg(
            (v_e_pure[k] / (rho2 * np.cos(np.deg2rad(lat_ins_pure[k])))) * dt
        )

        Z[k] = np.array([lat_ins[k] - lat, lon_ins[k] - lon, v_e_ins[k], v_n_ins[k]])

        C = transf

        F[k] = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [-omega_shu**2, 0, 0, 0, n_H, 0, 0, 0, 0, C[0, 0], C[0, 1], C[0, 2], 0],
                [0, -omega_shu**2, 0, 0, -n_H, 0, n_e, 0, 0, C[1, 0], C[1, 1], C[1, 2], 0],
                [0, 0, 0, 0, 0, 0, 0, C[0, 0], C[0, 1], C[0, 2], 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, C[1, 0], C[1, 1], C[1, 2], 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, C[2, 0], C[2, 1], C[2, 2], 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        Q[k] = build_Q_from_imu_noise(C, dt)

        sigma_lat = sigma_pos_m / rho1
        sigma_lon = sigma_pos_m / (rho2 * np.cos(np.deg2rad(lat_ins[k])))

        R = np.diag(
            [
                sigma_lon**2,
                sigma_lat**2,
                sigma_vel_mps**2,
                sigma_vel_mps**2,
            ]
        )

        H[k] = np.array(
            [
                [1 / (rho2 * np.cos(np.deg2rad(lat))), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1 / rho1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [
                    -(V_h / rho2 + omega[1] * np.tan(np.deg2rad(lat))),
                    -omega[2],
                    1,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                [-V_h / rho1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        )

        G[k] = build_G(C)
        Fi[k] = np.eye(13) + F[k] * dt
        Ge[k] = G[k] * dt

        S[k] = (Fi[k - 1] @ P[k - 1] @ np.transpose(Fi[k - 1])) + (Q[k - 1])
        K[k] = S[k] @ np.transpose(H[k]) @ np.linalg.inv(H[k] @ S[k] @ np.transpose(H[k]) + R)
        P[k] = (np.eye(13) - K[k] @ H[k]) @ S[k]

        X[k] = Fi[k - 1] @ X[k - 1] + K[k] @ (Z[k] - H[k] @ Fi[k - 1] @ X[k - 1])

        # Оценка после коррекции (используется строка состояния X[k], см. filterKalman.py)
        lon_kalm[k] = lon_ins[k] - X[k][0] / (rho2 * np.cos(np.deg2rad(lat_ins[k])))
        lat_kalm[k] = lat_ins[k] - X[k][1] / rho1

    gnss = list(zip(lat_meas.tolist(), lon_meas.tolist()))
    ins = list(zip(lat_ins.tolist(), lon_ins.tolist()))
    ins_pure = list(zip(lat_ins_pure.tolist(), lon_ins_pure.tolist()))
    kalman = [(la, lo) for la, lo in zip(lat_kalm.tolist(), lon_kalm.tolist()) if np.isfinite(la) and np.isfinite(lo)]

    return gnss, ins, ins_pure, kalman


def build_map(gnss, ins, ins_pure, kalman, path_out: Path) -> None:
    gnss = _finite_latlon(gnss)
    ins = _finite_latlon(ins)
    ins_pure = _finite_latlon(ins_pure)
    kalman = _finite_latlon(kalman)
    if not gnss:
        raise ValueError("Нет ни одной точки GNSS с конечными координатами для карты")

    lat0 = gnss[len(gnss) // 2][0]
    lon0 = gnss[len(gnss) // 2][1]
    m = folium.Map(location=[lat0, lon0], zoom_start=16, tiles="OpenStreetMap")

    fg_gnss = folium.FeatureGroup(name="GNSS (измерения)", show=True)
    fg_ins_pure = folium.FeatureGroup(
        name="ИНС без коррекций (чистая интеграция ускорений)",
        show=True,
    )
    fg_ins_loop = folium.FeatureGroup(
        name="ИНС в контуре фильтра (с вычетом оценок смещений IMU)",
        show=True,
    )
    fg_kalm = folium.FeatureGroup(name="После фильтра Калмана", show=True)

    folium.PolyLine(
        gnss,
        color="blue",
        weight=3,
        opacity=0.9,
        tooltip="GNSS",
    ).add_to(fg_gnss)
    if ins_pure:
        folium.PolyLine(
            ins_pure,
            color="darkred",
            weight=4,
            opacity=0.9,
            tooltip="ИНС без коррекций",
        ).add_to(fg_ins_pure)
    if ins:
        folium.PolyLine(
            ins,
            color="orange",
            weight=3,
            opacity=0.8,
            tooltip="ИНС (оценки смещений из фильтра)",
        ).add_to(fg_ins_loop)
    if kalman:
        folium.PolyLine(
            kalman,
            color="green",
            weight=3,
            opacity=0.85,
            tooltip="Калман",
        ).add_to(fg_kalm)

    fg_gnss.add_to(m)
    fg_ins_pure.add_to(m)
    fg_ins_loop.add_to(m)
    fg_kalm.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(path_out))


def main() -> None:
    if not FK_PATH.is_file():
        print(f"Не найден {FK_PATH}", file=sys.stderr)
        sys.exit(1)
    if not CSV_PATH.is_file():
        print(f"Не найден {CSV_PATH}", file=sys.stderr)
        sys.exit(1)

    fk = _load_filter_kalman_api()
    inputs = fk["parse_csv"](CSV_PATH)
    _forward_fill_imu(inputs)
    lat_meas, lon_meas = _forward_fill_coords(inputs)

    gnss, ins, ins_pure, kalman = run_kalman_track(fk, inputs, lat_meas, lon_meas)
    build_map(gnss, ins, ins_pure, kalman, OUT_HTML)
    print(f"Карта сохранена: {OUT_HTML}")


if __name__ == "__main__":
    main()
