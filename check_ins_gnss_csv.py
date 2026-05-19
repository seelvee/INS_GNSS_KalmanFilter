# -*- coding: utf-8 -*-
"""
Проверка CSV от arduino_ins_gnss.ino: загрузка, пересчёт ускорения в нав. СК, комплексирование, карта.

Алгоритмы:
- Парсинг CSV: колонки 0–12 (time_ms, lat, lon, alt_m, speed_mps, sats, fix, ax,ay,az, gx,gy,gz), опционально roll,pitch,yaw.
- Ускорение в нав. СК: из ax,ay,az (м/с²) и gx,gy,gz (град/с) — вычитаем GYRO_BIAS_DEG_PER_SEC, град→рад, затем BodyToNavProcessor (Пуассон + Родрига).
- Опорная точка: первый валидный ГНСС; скорость ГНСС в ENU по speed_mps и yaw_deg (v_E = speed*sin(yaw), v_N = speed*cos(yaw)).
- Интеграция «только ИНС» и комплексирование — та же модель, что в ins_gnss_filter (прогноз по acc_nav, коррекция по ГНСС).

Формат CSV от .ino: ax,ay,az в м/с², gx,gy,gz в град/с. Запуск: python check_ins_gnss_csv.py [путь_к_файлу.csv]
"""
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from geo_map import latlonalt_to_pos_enu, pos_enu_to_latlonalt, build_map_html
from ins_gnss_filter import FilterConfig, run_ins_gnss_complex, run_ins_gnss_hard_coupled
from imu_body_to_nav import BodyToNavProcessor, GYRO_BIAS_DEG_PER_SEC
from analyze_ins_csv import compute_acc_nav_from_imu, csv_diagnostics, detect_imu_units, imu_to_physical, get_scales_for_data


def parse_float(s):
    s = (s or "").strip().lower()
    if s in ("", "nan", "nan(ind)"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_ins_gnss_csv(path: str):
    """
    Загружает CSV. Базовые колонки 0–12: time_ms, lat, lon, alt_m, speed_mps, sats, fix, ax,ay,az, gx,gy,gz.
    Расширенный формат: при наличии yaw_deg в заголовке — roll_deg, pitch_deg, yaw_deg в 13–15 или 16–18.
    yaw_deg используется для построения вектора скорости ГНСС в ENU. Столбцы acc_nav_* не читаем.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError("В файле нет данных (или только комментарии)")
    header = lines[0].lower()
    header_parts = [h.strip().lower() for h in lines[0].split(",")]
    idx = {name: i for i, name in enumerate(header_parts)}
    has_extended = "yaw_deg" in header
    has_raw = "ax_raw" in idx and "gz_raw" in idx
    data_lines = lines[1:] if lines[0].startswith("time") or lines[0].startswith("t") else lines
    rows = []
    for line in data_lines:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 13:
            continue
        try:
            time_ms = int(parse_float(parts[0])) if parts[0] else 0
        except (ValueError, TypeError):
            time_ms = len(rows) * 100
        def v(name: str, default=float("nan")):
            i = idx.get(name, None)
            if i is None or i >= len(parts):
                return default
            return parse_float(parts[i])

        row = {
            "time_ms": time_ms,
            "lat": v("lat"),
            "lon": v("lon"),
            "alt_m": v("alt_m"),
            "speed_mps": v("speed_mps"),
            "fix": 1 if v("fix", 0.0) == 1 else 0,
            "ax": v("ax_raw") if has_raw else v("ax"),
            "ay": v("ay_raw") if has_raw else v("ay"),
            "az": v("az_raw") if has_raw else v("az"),
            "gx": v("gx_raw") if has_raw else v("gx"),
            "gy": v("gy_raw") if has_raw else v("gy"),
            "gz": v("gz_raw") if has_raw else v("gz"),
        }
        if has_extended:
            row["roll_deg"] = v("roll_deg", v("roll_dmp_deg"))
            row["pitch_deg"] = v("pitch_deg", v("pitch_dmp_deg"))
            row["yaw_deg"] = v("yaw_deg", v("yaw_dmp_deg"))
        rows.append(row)
    if not rows:
        raise ValueError("Нет строк с 13+ колонками")
    out = {
        "time_ms": np.array([r["time_ms"] for r in rows]),
        "lat": np.array([r["lat"] for r in rows]),
        "lon": np.array([r["lon"] for r in rows]),
        "alt_m": np.array([r["alt_m"] for r in rows]),
        "speed_mps": np.array([r["speed_mps"] for r in rows]),
        "fix": np.array([r["fix"] for r in rows], dtype=float),
        "ax": np.array([r["ax"] for r in rows]),
        "ay": np.array([r["ay"] for r in rows]),
        "az": np.array([r["az"] for r in rows]),
        "gx": np.array([r["gx"] for r in rows]),
        "gy": np.array([r["gy"] for r in rows]),
        "gz": np.array([r["gz"] for r in rows]),
    }
    if has_extended and "yaw_deg" in rows[0]:
        out["yaw_deg"] = np.array([r["yaw_deg"] for r in rows])
    else:
        out["yaw_deg"] = None
    out["imu_source"] = "dmp_raw" if has_raw else "physical"
    return out


def get_acc_nav_and_time(data: dict):
    """
    Возвращает (time_sec, acc_nav). Ускорение в нав. СК считаем только из ИМУ: ax,ay,az (м/с²), gx,gy,gz (град/с).
    Алгоритм: вычитаем калибровочный bias гироскопа (град/с), переводим в рад/с, затем compute_acc_nav_from_imu
    (BodyToNavProcessor — Пуассон, Родрига; a_nav = C @ f_body + g).
    """
    t_ms = data["time_ms"]
    time_sec = (t_ms - t_ms[0]) / 1000.0
    ax = np.asarray(data["ax"], dtype=float)
    ay = np.asarray(data["ay"], dtype=float)
    az = np.asarray(data["az"], dtype=float)
    gx = np.asarray(data["gx"], dtype=float)
    gy = np.asarray(data["gy"], dtype=float)
    gz = np.asarray(data["gz"], dtype=float)
    units, scales = detect_imu_units(ax, ay, az, gx, gy, gz)
    scales = get_scales_for_data(data, units, scales)
    ax, ay, az, gx_r, gy_r, gz_r = imu_to_physical(ax, ay, az, gx, gy, gz, units, scales)
    acc_nav, _ = compute_acc_nav_from_imu(
        t_ms, ax, ay, az, gx_r, gy_r, gz_r, initial_yaw=0.0,
    )
    return time_sec, acc_nav


def main():
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
    else:
        csv_path = input("Путь к CSV файлу: ").strip()
    if not csv_path or not os.path.isfile(csv_path):
        print("Файл не найден:", csv_path)
        sys.exit(1)
    csv_path = os.path.abspath(csv_path)
    print("Загрузка:", csv_path)
    data = load_ins_gnss_csv(csv_path)
    n = len(data["time_ms"])
    print(f"Строк: {n}")
    csv_diagnostics(data, expect_dt_ms=100.0)

    time_sec, acc_nav = get_acc_nav_and_time(data)
    valid = np.isfinite(data["lat"]) & np.isfinite(data["lon"])
    if not np.any(valid):
        print("Нет валидных координат ГНСС в файле.")
        sys.exit(1)
    i0 = np.where(valid)[0][0]
    lat0 = float(data["lat"][i0])
    lon0 = float(data["lon"][i0])
    alt0 = float(data["alt_m"][i0]) if np.isfinite(data["alt_m"][i0]) else 0.0
    print(f"Опорная точка: lat={lat0:.6f}, lon={lon0:.6f}, alt={alt0:.1f} м")

    pos_ll = np.column_stack([data["lat"], data["lon"], data["alt_m"]])
    pos_enu = latlonalt_to_pos_enu(pos_ll, lat0, lon0, alt0)
    gnss_pos = pos_enu.copy()
    gnss_pos[~valid, :] = np.nan
    speed = np.nan_to_num(data["speed_mps"], nan=0.0)
    yaw_deg = data["yaw_deg"]
    use_vel = yaw_deg is not None and np.isfinite(yaw_deg).any()
    if use_vel:
        yaw_rad = np.radians(np.nan_to_num(yaw_deg, nan=0.0))
        # Вектор скорости в ENU: v_E = speed*sin(yaw), v_N = speed*cos(yaw) (yaw от севера по часовой)
        gnss_vel = np.column_stack([
            speed * np.sin(yaw_rad),
            speed * np.cos(yaw_rad),
            np.zeros(n),
        ])
        gnss_vel[~valid, :] = np.nan
    else:
        gnss_vel = np.full((n, 3), np.nan)
    acc_nav = np.nan_to_num(acc_nav, nan=0.0)

    dt = float(np.median(np.diff(time_sec))) if n > 1 else 0.1
    config = FilterConfig(
        dt=dt,
        sigma_acc=0.2,
        sigma_gnss_pos=3.0,
        sigma_gnss_vel=0.5,
        use_gnss_velocity=use_vel,
    )
    x0 = np.concatenate([gnss_pos[i0], gnss_vel[i0] if np.isfinite(gnss_vel[i0]).all() else np.zeros(3)])
    if not np.isfinite(x0).all():
        x0 = np.concatenate([gnss_pos[i0], np.zeros(3)])

    # Интеграция «только ИНС»: v_i = v_{i-1} + a_i*dt, p_i = p_{i-1} + v*dt + 0.5*a*dt² (как в КФ), без коррекции по ГНСС
    pos_ins = np.zeros((n, 3))
    vel_ins = np.zeros((n, 3))
    pos_ins[0] = x0[:3]
    vel_ins[0] = x0[3:6]
    for i in range(1, n):
        dt_i = time_sec[i] - time_sec[i - 1]
        dt_i = np.clip(dt_i, 1e-6, 1.0)
        vel_ins[i] = vel_ins[i - 1] + acc_nav[i] * dt_i
        pos_ins[i] = pos_ins[i - 1] + vel_ins[i - 1] * dt_i + 0.5 * acc_nav[i] * (dt_i ** 2)
    ins_only_ll = pos_enu_to_latlonalt(pos_ins, lat0, lon0, alt0)

    # Комплексирование: фильтр Калмана — прогноз по acc_nav, коррекция по положению и (при наличии) скорости ГНСС
    positions, velocities, _ = run_ins_gnss_complex(
        time_grid=time_sec,
        acc_ins=acc_nav,
        pos_gnss=gnss_pos,
        vel_gnss=gnss_vel if config.use_gnss_velocity else None,
        config=config,
        x0=x0,
    )
    hard_positions, hard_velocities = run_ins_gnss_hard_coupled(
        time_grid=time_sec,
        acc_ins=acc_nav,
        pos_gnss=gnss_pos,
        vel_gnss=gnss_vel if use_vel else None,
        x0=x0,
    )
    combined_ll = pos_enu_to_latlonalt(positions, lat0, lon0, alt0)
    hard_ll = pos_enu_to_latlonalt(hard_positions, lat0, lon0, alt0)
    gnss_ll = np.column_stack([data["lat"], data["lon"], data["alt_m"]])
    gnss_ll[~valid, :] = np.nan
    out_dir = os.path.dirname(csv_path)
    map_path = os.path.join(out_dir, "ins_gnss_check_map.html")
    build_map_html(
        latlon_true=None,
        latlon_ins=ins_only_ll,
        latlon_combined=hard_ll,
        latlon_gnss=gnss_ll,
        output_path=map_path,
        title="Проверка CSV: ИНС+ГНСС (жесткая связь)",
    )
    print("Карта сохранена:", map_path)
    print("  Фиолетовый пунктир = только ИНС, красная = суммарная (ИНС+ГНСС), зелёная = ГНСС.")

    # Графики: широта, долгота, высота, скорость — ИНС vs суммарная vs ГНСС
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        t = time_sec

        ax = axes[0, 0]
        ax.plot(t, ins_only_ll[:, 0], "m--", label="Только ИНС", alpha=0.9)
        ax.plot(t, combined_ll[:, 0], "r-", label="Калман (ИНС+ГНСС)")
        ax.plot(t, hard_ll[:, 0], "b-", label="Жесткая связь ИНС<-ГНСС")
        ax.plot(t, data["lat"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Широта, °")
        ax.legend()
        ax.grid(True)
        ax.set_title("Широта")

        ax = axes[0, 1]
        ax.plot(t, ins_only_ll[:, 1], "m--", label="Только ИНС", alpha=0.9)
        ax.plot(t, combined_ll[:, 1], "r-", label="Калман (ИНС+ГНСС)")
        ax.plot(t, hard_ll[:, 1], "b-", label="Жесткая связь ИНС<-ГНСС")
        ax.plot(t, data["lon"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Долгота, °")
        ax.legend()
        ax.grid(True)
        ax.set_title("Долгота")

        ax = axes[1, 0]
        ax.plot(t, ins_only_ll[:, 2], "m--", label="Только ИНС", alpha=0.9)
        ax.plot(t, combined_ll[:, 2], "r-", label="Калман (ИНС+ГНСС)")
        ax.plot(t, hard_ll[:, 2], "b-", label="Жесткая связь ИНС<-ГНСС")
        ax.plot(t, data["alt_m"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Высота, м")
        ax.legend()
        ax.grid(True)
        ax.set_title("Высота")

        ax = axes[1, 1]
        speed_ins = np.linalg.norm(vel_ins, axis=1)
        speed_combined = np.linalg.norm(velocities, axis=1)
        speed_hard = np.linalg.norm(hard_velocities, axis=1)
        ax.plot(t, speed_ins, "m--", label="Только ИНС", alpha=0.9)
        ax.plot(t, speed_combined, "r-", label="Калман (ИНС+ГНСС)")
        ax.plot(t, speed_hard, "b-", label="Жесткая связь ИНС<-ГНСС")
        ax.plot(t, data["speed_mps"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        ax.set_xlabel("Время, с")
        ax.set_ylabel("Скорость, м/с")
        ax.legend()
        ax.grid(True)
        ax.set_title("Скорость")

        plt.suptitle("Проверка CSV: траектория и скорость")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "ins_gnss_check_plots.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print("Графики сохранены:", plot_path)
    except Exception as e:
        print("Графики не построены:", e)


if __name__ == "__main__":
    main()
