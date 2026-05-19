# -*- coding: utf-8 -*-
"""
Анализ исходного CSV и полный пересчёт ИНС с нуля.

Алгоритмы:
1. Парсинг CSV: фиксированные индексы колонок (0–12 базовые, опционально roll/pitch/yaw).
2. Определение единиц ИМУ (detect_imu_units): по порогам |ax|,|gx| различаем LSB и физические величины.
3. Приведение к физическим единицам: LSB → raw_to_physical (масштабы MPU6050); physical — только bias гироскопа и град→рад.
4. Ускорение в нав. СК: уравнение Пуассона (BodyToNavProcessor), интегрирование Родрига за шаг dt, a_nav = C @ f_body + g.
5. Интеграция «только ИНС»: та же дискретная модель, что в КФ (v += a*dt, p += v*dt + 0.5*a*dt²).
6. Комплексирование: Калман (прогноз по ИНС, коррекция по ГНСС), опорная точка — первый валидный ГНСС.
7. Диагностика причин плохой ИНС: единицы, пропуски ИМУ, вертикальная компонента, дрейф скорости, bias гироскопа.

Запуск: python analyze_ins_csv.py путь_к_файлу.csv
"""
import math
import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from imu_body_to_nav import (
    ArduinoIMUScales,
    BodyToNavProcessor,
    raw_to_physical,
    GRAVITY,
    GYRO_BIAS_DEG_PER_SEC,
)
from ins_gnss_filter import FilterConfig, run_ins_gnss_complex, run_ins_gnss_hard_coupled
from geo_map import latlonalt_to_pos_enu, pos_enu_to_latlonalt, build_map_html


# ---- Парсер CSV (совместим с check_ins_gnss_csv и serial_to_file) ----
def parse_float(s):
    s = (s or "").strip().lower()
    if s in ("", "nan", "nan(ind)"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_csv(path: str):
    """
    Загружает CSV. Формат: колонки 0–12 — time_ms, lat, lon, alt_m, speed_mps, sats, fix, ax,ay,az, gx,gy,gz.
    Расширенный формат: наличие yaw_deg в заголовке; тогда roll,pitch,yaw в индексах 13–15 (16 колонок)
    или 16–18 (19 колонок, старый формат с acc_nav). Столбцы acc_nav_* не читаем — ускорение в нав. СК считаем по ИМУ.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]
    if not lines:
        raise ValueError("В файле нет данных")
    header = lines[0].lower()
    header_parts = [h.strip().lower() for h in lines[0].split(",")]
    idx = {name: i for i, name in enumerate(header_parts)}
    has_extended = "yaw_deg" in header
    data_lines = lines[1:] if lines[0].startswith("time") or lines[0].startswith("t") else lines
    has_raw = "ax_raw" in idx and "gz_raw" in idx
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


def csv_diagnostics(data: dict, expect_dt_ms: float = 100.0) -> None:
    """
    Диагностика загруженного CSV: медиана/мин/макс шага по времени, число валидных ГНСС (lat/lon, fix),
    число валидных строк ИМУ. expect_dt_ms — ожидаемый интервал вывода (например 100 мс для Arduino).
    """
    n = len(data["time_ms"])
    t_ms = data["time_ms"]
    dt_ms = np.diff(t_ms, prepend=t_ms[0])
    dt_ms = dt_ms[dt_ms > 0]
    if len(dt_ms) > 0:
        med_dt = float(np.median(dt_ms))
        min_dt = float(np.min(dt_ms))
        max_dt = float(np.max(dt_ms))
        print("\n--- Диагностика: шаг по времени ---")
        print("  Медиана dt: {:.0f} мс, мин: {:.0f}, макс: {:.0f}".format(med_dt, min_dt, max_dt))
        if abs(med_dt - expect_dt_ms) > 50:
            print("  Внимание: шаг сильно отличается от ожидаемых {:.0f} мс.".format(expect_dt_ms))

    valid_gnss = np.isfinite(data["lat"]) & np.isfinite(data["lon"])
    n_valid = int(np.sum(valid_gnss))
    n_fix1 = int(np.sum(data["fix"] == 1))
    print("--- Диагностика: ГНСС ---")
    print("  Строк с валидными lat/lon: {} из {} ({:.1f}%)".format(n_valid, n, 100.0 * n_valid / n if n else 0))
    print("  Строк с fix=1: {} из {}".format(n_fix1, n))
    if n_valid == 0:
        print("  Внимание: нет валидных координат ГНСС — комплексирование по положению невозможно.")

    valid_imu = np.isfinite(data["ax"]) & np.isfinite(data["gx"])
    n_imu = int(np.sum(valid_imu))
    print("--- Диагностика: ИМУ ---")
    print("  Строк с валидными ax,ay,az,gx,gy,gz: {} из {}".format(n_imu, n))
    if n_imu == 0:
        print("  Внимание: нет валидных ИМУ — ускорение в нав. СК будет нулевым.")


def detect_imu_units(ax, ay, az, gx, gy, gz):
    """
    Эвристика единиц измерения ИМУ по максимальным по модулю значениям.

    Алгоритм: акселерометр в м/с² редко даёт |a| > 100; в LSB (MPU6050 ±4g) — тысячи.
    Гироскоп в град/с редко > 500; в LSB — десятки тысяч. Пороги: a_max > 100 или g_max > 500 → LSB.
    Возвращает ("LSB", ArduinoIMUScales()) или ("physical", None). physical: ax,ay,az в м/с², gx,gy,gz в град/с.
    """
    a_flat = np.concatenate([np.ravel(ax), np.ravel(ay), np.ravel(az)])
    g_flat = np.concatenate([np.ravel(gx), np.ravel(gy), np.ravel(gz)])
    a_flat = a_flat[np.isfinite(a_flat)]
    g_flat = g_flat[np.isfinite(g_flat)]

    if len(a_flat) == 0 or len(g_flat) == 0:
        return "physical", None

    a_max = np.max(np.abs(a_flat))
    g_max = np.max(np.abs(g_flat))

    # Пороги: если числа большие — скорее всего LSB
    acc_looks_lsb = a_max > 100   # м/с² редко > 100 по модулю в типичных данных
    gyro_looks_lsb = g_max > 500  # град/с редко > 500

    if acc_looks_lsb or gyro_looks_lsb:
        return "LSB", ArduinoIMUScales()
    return "physical", None


def imu_to_physical(ax, ay, az, gx, gy, gz, units: str, scales: ArduinoIMUScales):
    """
    Приведение ИМУ к физическим единицам: ax,ay,az → м/с², gx,gy,gz → рад/с.
    LSB: raw_to_physical (масштабы и смещения MPU6050). physical: вычитаем GYRO_BIAS_DEG_PER_SEC, град→рад.
    """
    ax, ay, az = np.asarray(ax), np.asarray(ay), np.asarray(az)
    gx, gy, gz = np.asarray(gx), np.asarray(gy), np.asarray(gz)

    if units == "LSB" and scales is not None:
        raw_acc = np.column_stack([ax, ay, az])
        raw_gyro = np.column_stack([gx, gy, gz])
        acc_body = np.zeros_like(raw_acc)
        gyro_rad = np.zeros_like(raw_gyro)
        for i in range(len(ax)):
            a, g = raw_to_physical(
                raw_acc[i], raw_gyro[i], scales
            )
            acc_body[i] = a
            gyro_rad[i] = g
        return acc_body[:, 0], acc_body[:, 1], acc_body[:, 2], gyro_rad[:, 0], gyro_rad[:, 1], gyro_rad[:, 2]

    # physical: ax,ay,az м/с², gx,gy,gz град/с; вычитаем калибровочный bias гироскопа (град/с) → рад/с
    bx, by, bz = GYRO_BIAS_DEG_PER_SEC
    gx_r = np.deg2rad(np.nan_to_num(gx, nan=0.0) - bx)
    gy_r = np.deg2rad(np.nan_to_num(gy, nan=0.0) - by)
    gz_r = np.deg2rad(np.nan_to_num(gz, nan=0.0) - bz)
    return ax, ay, az, gx_r, gy_r, gz_r


def get_scales_for_data(data: dict, units: str, auto_scales: ArduinoIMUScales):
    """
    Выбор масштабов для raw-данных.
    Для DMP raw (ax_raw..gz_raw): считаем, что MPU6050 в конфиге DMP по умолчанию
    отдаёт accel ±2g (16384 LSB/g) и gyro ±250 deg/s (131 LSB/(deg/s)).
    """
    if units != "LSB":
        return auto_scales
    if data.get("imu_source") == "dmp_raw":
        return ArduinoIMUScales(
            accel_scale=9.80665 / 16384.0,
            gyro_scale=(np.pi / 180.0) / 131.0,
            accel_bias=(0.0, 0.0, 0.0),
            gyro_bias=(0.0, 0.0, 0.0),
        )
    return auto_scales


def compute_acc_nav_from_imu(
    time_ms, ax, ay, az, gx_rad, gy_rad, gz_rad,
    initial_yaw=0.0,
):
    """
    Ускорение в навигационной СК (ENU) по показаниям ИМУ в связанной СК.

    Алгоритм: BodyToNavProcessor — уравнение Пуассона (кинематика вращения), интегрирование за шаг dt
    по формуле Родрига: C_{k+1} = C_k @ exp([ω×] dt). Затем a_nav = C @ f_body + g_nav (g_nav = (0,0,-g)).
    Гироскоп уже в рад/с, акселерометр в м/с² (удельная сила в body). Возвращает acc_nav (N,3), м/с².
    """
    n = len(time_ms)
    dt_ms = np.diff(time_ms, prepend=time_ms[0])
    dt_sec = np.clip(dt_ms / 1000.0, 0.001, 1.0)

    processor = BodyToNavProcessor(initial_yaw=initial_yaw)
    acc_nav = np.zeros((n, 3))

    for i in range(n):
        if math.isnan(ax[i]) or math.isnan(gx_rad[i]):
            acc_nav[i] = np.nan
            continue
        processor.update_orientation(gx_rad[i], gy_rad[i], gz_rad[i], float(dt_sec[i]))
        acc_nav[i] = processor.accel_body_to_nav([ax[i], ay[i], az[i]])

    return acc_nav, dt_sec


def integrate_ins_only(time_sec, acc_nav, p0, v0):
    """
    Интеграция «только ИНС»: v_i = v_{i-1} + a_i*dt, p_i = p_{i-1} + v_{i-1}*dt + 0.5*a_i*dt².
    Та же дискретная модель, что в фильтре Калмана (прогноз по ускорению). NaN в acc_nav обнуляются.
    p0, v0 — начальные положение и скорость в ENU, м и м/с.
    """
    acc_nav = np.nan_to_num(acc_nav, nan=0.0)
    n = len(time_sec)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    pos[0] = p0
    vel[0] = v0
    for i in range(1, n):
        dt = time_sec[i] - time_sec[i - 1]
        dt = np.clip(dt, 1e-6, 1.0)
        vel[i] = vel[i - 1] + acc_nav[i] * dt
        pos[i] = pos[i - 1] + vel[i - 1] * dt + 0.5 * acc_nav[i] * (dt ** 2)
    return pos, vel


def main():
    if len(sys.argv) < 2:
        csv_path = input("Путь к CSV: ").strip()
    else:
        csv_path = sys.argv[1]
    if not csv_path or not os.path.isfile(csv_path):
        print("Файл не найден:", csv_path)
        sys.exit(1)
    csv_path = os.path.abspath(csv_path)

    print("=" * 60)
    print("АНАЛИЗ ИСХОДНОГО CSV И ПЕРЕСЧЁТ ИНС")
    print("=" * 60)
    print("Файл:", csv_path)

    data = load_csv(csv_path)
    n = len(data["time_ms"])
    print(f"Строк данных: {n}")

    csv_diagnostics(data, expect_dt_ms=100.0)

    # ---- 1) Сырые данные ИМУ из CSV (до приведения единиц) ----
    print("\n--- 1) Данные, приходящие в парсер (колонки ИМУ) ---")
    ax, ay, az = data["ax"], data["ay"], data["az"]
    gx, gy, gz = data["gx"], data["gy"], data["gz"]
    valid_imu = np.isfinite(ax) & np.isfinite(gx)
    if np.any(valid_imu):
        ax_v, ay_v, az_v = ax[valid_imu], ay[valid_imu], az[valid_imu]
        gx_v, gy_v, gz_v = gx[valid_imu], gy[valid_imu], gz[valid_imu]
        print(f"  ax: min={np.nanmin(ax_v):.2f}, max={np.nanmax(ax_v):.2f}, mean={np.nanmean(ax_v):.2f}")
        print(f"  ay: min={np.nanmin(ay_v):.2f}, max={np.nanmax(ay_v):.2f}")
        print(f"  az: min={np.nanmin(az_v):.2f}, max={np.nanmax(az_v):.2f}")
        print(f"  gx: min={np.nanmin(gx_v):.2f}, max={np.nanmax(gx_v):.2f}")
        print(f"  gy: min={np.nanmin(gy_v):.2f}, max={np.nanmax(gy_v):.2f}")
        print(f"  gz: min={np.nanmin(gz_v):.2f}, max={np.nanmax(gz_v):.2f}")
    else:
        print("  Нет валидных ИМУ.")

    # ---- 2) Определение единиц ----
    units, scales = detect_imu_units(ax, ay, az, gx, gy, gz)
    scales = get_scales_for_data(data, units, scales)
    print("\n--- 2) Определение единиц измерения ИМУ ---")
    if units == "LSB":
        print("  Решение: данные в LSB (сырые с датчика). Будет применён перевод в м/с² и рад/с (MPU6050).")
    else:
        print("  Решение: данные уже в физических единицах (ax,ay,az — м/с², gx,gy,gz — град/с).")

    ax_p, ay_p, az_p, gx_r, gy_r, gz_r = imu_to_physical(
        ax, ay, az, gx, gy, gz, units, scales
    )
    if np.any(np.isfinite(ax_p)):
        print("  После приведения: |a|_max ≈ {:.2f} м/с², |ω|_max ≈ {:.4f} рад/с".format(
            np.nanmax(np.sqrt(ax_p**2 + ay_p**2 + az_p**2)),
            np.nanmax(np.sqrt(gx_r**2 + gy_r**2 + gz_r**2))
        ))

    # ---- 3) Пересчёт ускорения в нав. СК (Пуассон + Родрига, BodyToNavProcessor) ----
    time_sec = (data["time_ms"] - data["time_ms"][0]) / 1000.0
    acc_nav, dt_sec = compute_acc_nav_from_imu(
        data["time_ms"], ax_p, ay_p, az_p, gx_r, gy_r, gz_r, initial_yaw=0.0
    )
    acc_nav_clean = np.nan_to_num(acc_nav, nan=0.0)
    print("\n--- 3) Ускорение в навигационной СК (ENU) ---")
    print("  |acc_nav|: min={:.3f}, max={:.3f} м/с²".format(
        np.min(np.linalg.norm(acc_nav_clean, axis=1)),
        np.max(np.linalg.norm(acc_nav_clean, axis=1))
    ))
    # В покое вертикальная компонента должна быть ~0 (после вычета g)
    up = acc_nav_clean[:, 2]
    print("  acc_nav[Up]: mean={:.3f} (ожидается ~0 в покое)".format(np.mean(up)))

    # ---- 4) Опорная точка ENU и начальная скорость (первый валидный ГНСС; при наличии yaw — скорость по курсу) ----
    valid = np.isfinite(data["lat"]) & np.isfinite(data["lon"])
    if not np.any(valid):
        print("Нет валидных координат ГНСС. Дальнейший расчёт по первой точке (0,0,0).")
        i0 = 0
        lat0, lon0, alt0 = 0.0, 0.0, 0.0
        p0_enu = np.zeros(3)
        v0_enu = np.zeros(3)
    else:
        i0 = np.where(valid)[0][0]
        lat0 = float(data["lat"][i0])
        lon0 = float(data["lon"][i0])
        alt0 = float(data["alt_m"][i0]) if np.isfinite(data["alt_m"][i0]) else 0.0
        pos_ll = np.column_stack([data["lat"], data["lon"], data["alt_m"]])
        pos_enu_all = latlonalt_to_pos_enu(pos_ll, lat0, lon0, alt0)
        p0_enu = pos_enu_all[i0]
        speed = np.nan_to_num(data["speed_mps"], nan=0.0)
        yaw_deg = data["yaw_deg"]
        if yaw_deg is not None and np.isfinite(yaw_deg[i0]):
            yaw_rad = np.radians(yaw_deg[i0])
            v0_enu = np.array([
                speed[i0] * math.sin(yaw_rad),
                speed[i0] * math.cos(yaw_rad),
                0.0,
            ])
        else:
            v0_enu = np.zeros(3)
        print("\n--- 4) Опорная точка (первый валидный ГНСС) ---")
        print("  lat0={:.6f}, lon0={:.6f}, alt0={:.1f} м".format(lat0, lon0, alt0))
        print("  p0_enu =", p0_enu)
        print("  v0_enu =", v0_enu)

    # ---- 5) Интеграция «только ИНС» (без коррекции ГНСС) ----
    pos_ins, vel_ins = integrate_ins_only(time_sec, acc_nav, p0_enu, v0_enu)
    ins_only_ll = pos_enu_to_latlonalt(pos_ins, lat0, lon0, alt0)

    # ---- 6) Комплексирование ИНС+ГНСС (фильтр Калмана: прогноз по acc_nav, коррекция по pos/vel ГНСС) ----
    pos_ll = np.column_stack([data["lat"], data["lon"], data["alt_m"]])
    pos_enu_gnss = latlonalt_to_pos_enu(pos_ll, lat0, lon0, alt0)
    gnss_pos = pos_enu_gnss.copy()
    gnss_pos[~valid, :] = np.nan
    gnss_vel = np.full((n, 3), np.nan)
    if data["yaw_deg"] is not None and np.isfinite(data["yaw_deg"]).any():
        yaw_rad = np.radians(np.nan_to_num(data["yaw_deg"], nan=0.0))
        speed = np.nan_to_num(data["speed_mps"], nan=0.0)
        gnss_vel = np.column_stack([
            speed * np.sin(yaw_rad),
            speed * np.cos(yaw_rad),
            np.zeros(n),
        ])
        gnss_vel[~valid, :] = np.nan
        use_vel = True
    else:
        use_vel = False

    dt_med = float(np.median(np.diff(time_sec))) if n > 1 else 0.1
    config = FilterConfig(
        dt=dt_med,
        sigma_acc=0.2,
        sigma_gnss_pos=3.0,
        sigma_gnss_vel=0.5,
        use_gnss_velocity=use_vel,
    )
    x0 = np.concatenate([p0_enu, v0_enu])
    positions, velocities, _ = run_ins_gnss_complex(
        time_grid=time_sec,
        acc_ins=acc_nav_clean,
        pos_gnss=gnss_pos,
        vel_gnss=gnss_vel if use_vel else None,
        config=config,
        x0=x0,
    )
    hard_positions, hard_velocities = run_ins_gnss_hard_coupled(
        time_grid=time_sec,
        acc_ins=acc_nav_clean,
        pos_gnss=gnss_pos,
        vel_gnss=gnss_vel if use_vel else None,
        x0=x0,
    )
    combined_ll = pos_enu_to_latlonalt(positions, lat0, lon0, alt0)
    hard_ll = pos_enu_to_latlonalt(hard_positions, lat0, lon0, alt0)

    # ---- 7) Диагностика причин плохой работы ИНС (единицы, пропуски, bias, дрейф) ----
    print("\n" + "=" * 60)
    print("ПОЧЕМУ ИНС МОЖЕТ ДАВАТЬ ПЛОХИЕ ДАННЫЕ")
    print("=" * 60)
    reasons = []
    if units == "physical" and np.nanmax(np.abs(ax)) > 100:
        reasons.append("Единицы ИМУ: в CSV большие числа (ax,ay,az > 100), но интерпретация — «м/с²». Возможно, в файле записаны LSB — тогда ускорение и ориентация считаются неверно. Нужно либо записывать в CSV уже м/с² и град/с, либо включать автоопределение LSB (как в этом скрипте).")
    elif units == "LSB":
        reasons.append("Данные были в LSB; пересчёт выполнен с масштабами MPU6050. Если датчик другой — нужны свои масштабы и смещения нуля.")
    if not np.any(valid_imu):
        reasons.append("В части строк ИМУ отсутствуют (NaN) — подставляется нулевое ускорение, что искажает интеграцию.")
    if np.mean(np.abs(up)) > 1.0:
        reasons.append("Среднее вертикальное ускорение в нав. СК сильно отличается от 0 — возможен смещение акселерометра (bias) или ошибка ориентации из-за дрейфа гироскопа.")
    # Дрейф скорости ИНС
    speed_ins = np.linalg.norm(vel_ins, axis=1)
    if n > 10 and np.isfinite(speed_ins).all():
        drift_end = np.mean(speed_ins[-min(n//4, 50):]) - np.mean(speed_ins[:min(n//4, 50)])
        if abs(drift_end) > 0.5:
            reasons.append("Наблюдается дрейф скорости по чистой ИНС (типично из-за смещения акселерометра и/или гироскопа). Требуется калибровка нуля.")
    reasons.append("Курс в начале принят 0 и интегрируется только по гироскопу — любой постоянный bias гироскопа даёт линейный рост ошибки курса и искажение перевода ускорения в нав. СК.")
    for i, r in enumerate(reasons, 1):
        print("  {}. {}".format(i, r))

    # ---- 8) Визуализация: карта (folium) и графики широта/долгота/высота/скорость ----
    out_dir = os.path.dirname(csv_path)
    gnss_ll = np.column_stack([data["lat"], data["lon"], data["alt_m"]])
    gnss_ll[~valid, :] = np.nan
    map_path = os.path.join(out_dir, "ins_gnss_analysis_map.html")
    build_map_html(
        latlon_true=None,
        latlon_ins=ins_only_ll,
        latlon_combined=hard_ll,
        latlon_gnss=gnss_ll,
        output_path=map_path,
        title="Анализ CSV: ИНС (пересчёт) + ГНСС (жесткая связь)",
    )
    print("\nКарта сохранена:", map_path)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        t = time_sec

        axes[0, 0].plot(t, ins_only_ll[:, 0], "m--", label="Только ИНС", alpha=0.9)
        axes[0, 0].plot(t, combined_ll[:, 0], "r-", label="Калман (ИНС+ГНСС)")
        axes[0, 0].plot(t, hard_ll[:, 0], "b-", label="Жесткая связь ИНС<-ГНСС")
        axes[0, 0].plot(t, data["lat"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        axes[0, 0].set_xlabel("Время, с")
        axes[0, 0].set_ylabel("Широта, °")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_title("Широта")

        axes[0, 1].plot(t, ins_only_ll[:, 1], "m--", label="Только ИНС", alpha=0.9)
        axes[0, 1].plot(t, combined_ll[:, 1], "r-", label="Калман (ИНС+ГНСС)")
        axes[0, 1].plot(t, hard_ll[:, 1], "b-", label="Жесткая связь ИНС<-ГНСС")
        axes[0, 1].plot(t, data["lon"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        axes[0, 1].set_xlabel("Время, с")
        axes[0, 1].set_ylabel("Долгота, °")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        axes[0, 1].set_title("Долгота")

        axes[1, 0].plot(t, ins_only_ll[:, 2], "m--", label="Только ИНС", alpha=0.9)
        axes[1, 0].plot(t, combined_ll[:, 2], "r-", label="Калман (ИНС+ГНСС)")
        axes[1, 0].plot(t, hard_ll[:, 2], "b-", label="Жесткая связь ИНС<-ГНСС")
        axes[1, 0].plot(t, data["alt_m"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        axes[1, 0].set_xlabel("Время, с")
        axes[1, 0].set_ylabel("Высота, м")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_title("Высота")

        speed_comb = np.linalg.norm(velocities, axis=1)
        speed_hard = np.linalg.norm(hard_velocities, axis=1)
        axes[1, 1].plot(t, speed_ins, "m--", label="Только ИНС", alpha=0.9)
        axes[1, 1].plot(t, speed_comb, "r-", label="Калман (ИНС+ГНСС)")
        axes[1, 1].plot(t, speed_hard, "b-", label="Жесткая связь ИНС<-ГНСС")
        axes[1, 1].plot(t, data["speed_mps"], "g.", markersize=2, alpha=0.7, label="ГНСС")
        axes[1, 1].set_xlabel("Время, с")
        axes[1, 1].set_ylabel("Скорость, м/с")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_title("Скорость")

        plt.suptitle("Анализ CSV: пересчёт ИНС с учётом единиц (LSB/physical)")
        plt.tight_layout()
        plot_path = os.path.join(out_dir, "ins_gnss_analysis_plots.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print("Графики сохранены:", plot_path)
    except Exception as e:
        print("Графики не построены:", e)

    print("\nГотово.")


if __name__ == "__main__":
    main()
