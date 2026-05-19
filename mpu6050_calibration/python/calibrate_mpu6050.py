#!/usr/bin/env python3
"""
CLI для калибровки MPU6050 через Arduino-шлюз mpu6050_stream.ino.

Режимы:
  stationary    — неподвижный датчик, оценка gyro bias и шумовых ковариаций
  six_position  — калибровка акселерометра по шести ориентациям ±X, ±Y, ±Z

Примеры:
  python calibrate_mpu6050.py --port COM5 --baud 115200 --mode stationary --duration 300
  python calibrate_mpu6050.py --port COM5 --mode six_position --pose-duration 20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import serial
    from serial import SerialException
except ImportError as e:
    raise SystemExit("Установите pyserial: pip install pyserial") from e

from imu_utils import (
    ACCEL_LSB_PER_G,
    GYRO_LSB_PER_DPS,
    G_MPS2,
    SixPoseMeans,
    accel_bias_scale_from_six_poses,
    calibrated_accel_mps2,
    covariance_matrix,
    ensure_dir,
    estimate_sample_rate_hz,
    gyro_raw_to_rad_s,
    accel_raw_to_mps2,
    iso_now,
    load_raw_csv,
    magnitude_mps2,
    noise_std_per_axis,
    parse_csv_line,
    plot_histograms,
    plot_timeseries,
    write_json,
)


MIN_SAMPLES_STATIONARY = 200
MIN_SAMPLES_POSE = 400


def flush_serial(ser: serial.Serial) -> None:
    ser.reset_input_buffer()
    ser.reset_output_buffer()


def send_command(ser: serial.Serial, cmd: str) -> None:
    ser.write((cmd.strip() + "\n").encode("ascii"))


def read_lines_until(
    ser: serial.Serial,
    predicate,
    timeout_s: float = 3.0,
    max_lines: int = 5000,
) -> List[str]:
    """Читает строки до выполнения predicate(line) или таймаута."""
    deadline = time.time() + timeout_s
    buf: List[str] = []
    while time.time() < deadline and len(buf) < max_lines:
        raw = ser.readline()
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        buf.append(line)
        if predicate(line):
            return buf
    raise TimeoutError("Таймаут ожидания ответа MPU/Arduino")


def wait_for_comment(ser: serial.Serial, token: str, timeout_s: float = 5.0) -> None:
    token_u = token.upper()

    def ok(line: str) -> bool:
        return token_u in line.upper()

    read_lines_until(ser, ok, timeout_s=timeout_s)


def collect_csv_stream(
    ser: serial.Serial,
    duration_s: float,
    silence_timeout_s: float = 2.5,
) -> pd.DataFrame:
    """
    После команды START_STREAM читает строки CSV заданное время.
    При отсутствии валидных данных дольше silence_timeout_s — ошибка соединения.
    """
    flush_serial(ser)
    send_command(ser, "STOP_STREAM")
    time.sleep(0.05)
    flush_serial(ser)

    send_command(ser, "START_STREAM")
    wait_for_comment(ser, "STREAM_START", timeout_s=5.0)

    t_end = time.time() + duration_s
    rows = []
    last_sample = time.time()

    while time.time() < t_end:
        raw = ser.readline()
        if not raw:
            if time.time() - last_sample > silence_timeout_s:
                send_command(ser, "STOP_STREAM")
                raise ConnectionError(
                    f"Нет данных более {silence_timeout_s} с — проверьте кабель и прошивку."
                )
            continue

        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line or line.startswith("#"):
            continue

        parsed = parse_csv_line(line)
        if parsed is None:
            continue

        t_ms, vec = parsed
        rows.append(
            {
                "time_ms": int(t_ms),
                "ax_raw": int(vec[0]),
                "ay_raw": int(vec[1]),
                "az_raw": int(vec[2]),
                "gx_raw": int(vec[3]),
                "gy_raw": int(vec[4]),
                "gz_raw": int(vec[5]),
                "temp_raw": int(vec[6]),
            }
        )
        last_sample = time.time()

    send_command(ser, "STOP_STREAM")
    time.sleep(0.05)

    if len(rows) < 10:
        raise ValueError("Слишком мало валидных строк CSV — проверьте формат потока.")

    df = pd.DataFrame(rows)
    df.sort_values("time_ms", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def stationary_calibration(df: pd.DataFrame, duration_s: float) -> Tuple[Dict, pd.DataFrame]:
    """
    Статика на месте:
      - gyro bias = mean(raw gx,gy,gz)
      - gyro noise после вычитания bias → std и cov (в rad/s)
      - аксель в м/с²: вычитаем среднее по каждой оси (удельная сила≈const), шум — остаток
    """
    n = len(df)
    if n < MIN_SAMPLES_STATIONARY:
        raise ValueError(f"Слишком мало измерений ({n}). Нужно ≥ {MIN_SAMPLES_STATIONARY}.")

    gyro_raw = df[["gx_raw", "gy_raw", "gz_raw"]].to_numpy(dtype=np.float64)
    accel_raw = df[["ax_raw", "ay_raw", "az_raw"]].to_numpy(dtype=np.float64)

    gyro_bias_raw = np.mean(gyro_raw, axis=0)
    gyro_rad = gyro_raw_to_rad_s(gyro_raw)
    gyro_bias_rad = gyro_raw_to_rad_s(gyro_bias_raw.reshape(1, 3)).reshape(3)
    gyro_centered_rad = gyro_rad - gyro_bias_rad.reshape(1, 3)

    accel_mps2 = accel_raw_to_mps2(accel_raw)
    accel_mean = np.mean(accel_mps2, axis=0)
    accel_centered = accel_mps2 - accel_mean.reshape(1, 3)

    gyro_noise_std = noise_std_per_axis(gyro_centered_rad)
    accel_noise_std = noise_std_per_axis(accel_centered)

    gyro_cov = covariance_matrix(gyro_centered_rad)
    accel_cov = covariance_matrix(accel_centered)

    stats = {
        "duration_s": duration_s,
        "n_samples": int(n),
        "mean_ax_raw": float(np.mean(accel_raw[:, 0])),
        "mean_ay_raw": float(np.mean(accel_raw[:, 1])),
        "mean_az_raw": float(np.mean(accel_raw[:, 2])),
        "std_ax_raw": float(np.std(accel_raw[:, 0], ddof=1)),
        "std_ay_raw": float(np.std(accel_raw[:, 1], ddof=1)),
        "std_az_raw": float(np.std(accel_raw[:, 2], ddof=1)),
        "mean_gx_raw": float(gyro_bias_raw[0]),
        "mean_gy_raw": float(gyro_bias_raw[1]),
        "mean_gz_raw": float(gyro_bias_raw[2]),
        "gyro_bias_rad_s": gyro_bias_rad.tolist(),
        "gyro_noise_std_rad_s": gyro_noise_std.tolist(),
        "accel_noise_std_mps2": accel_noise_std.tolist(),
        "gyro_cov_rad_s2": gyro_cov.tolist(),
        "accel_cov_mps2": accel_cov.tolist(),
        "sample_rate_hz_estimated": float(estimate_sample_rate_hz(df["time_ms"].to_numpy())),
    }

    report_rows = []
    for k, v in stats.items():
        if isinstance(v, list):
            continue
        report_rows.append({"parameter": k, "value": v})

    report_df = pd.DataFrame(report_rows)
    return stats, report_df


def six_position_calibration(
    ser: serial.Serial,
    pose_duration_s: float,
    raw_chunks: List[pd.DataFrame],
) -> Tuple[SixPoseMeans, pd.DataFrame]:
    poses = [
        ("+X вверх", "plus_x"),
        ("-X вверх", "minus_x"),
        ("+Y вверх", "plus_y"),
        ("-Y вверх", "minus_y"),
        ("+Z вверх", "plus_z"),
        ("-Z вверх", "minus_z"),
    ]

    means: Dict[str, np.ndarray] = {}

    for label, key in poses:
        input(f"\nУстановите датчик: {label}. Затем нажмите Enter для сбора ~{pose_duration_s} с...")
        df_pose = collect_csv_stream(ser, duration_s=pose_duration_s)
        raw_chunks.append(df_pose.copy())

        if len(df_pose) < MIN_SAMPLES_POSE:
            raise ValueError(f"Мало данных в позе '{label}' ({len(df_pose)}).")

        m = df_pose[["ax_raw", "ay_raw", "az_raw"]].mean().to_numpy(dtype=np.float64)
        means[key] = m
        print(f"  Среднее raw для {label}: ax={m[0]:.1f}, ay={m[1]:.1f}, az={m[2]:.1f}")

    sm = SixPoseMeans(
        plus_x=means["plus_x"],
        minus_x=means["minus_x"],
        plus_y=means["plus_y"],
        minus_y=means["minus_y"],
        plus_z=means["plus_z"],
        minus_z=means["minus_z"],
    )

    rows = []
    for label, key in poses:
        m = means[key]
        rows.append({"pose": label, "mean_ax_raw": m[0], "mean_ay_raw": m[1], "mean_az_raw": m[2]})

    report_poses = pd.DataFrame(rows)
    return sm, report_poses


def verify_six_poses(
    means: SixPoseMeans,
    bias: np.ndarray,
    scale: np.ndarray,
) -> pd.DataFrame:
    """Проверка: модуль откалиброванного ускорения в каждой позе ~ g."""
    mats = [
        ("+X", means.plus_x.reshape(1, 3)),
        ("-X", means.minus_x.reshape(1, 3)),
        ("+Y", means.plus_y.reshape(1, 3)),
        ("-Y", means.minus_y.reshape(1, 3)),
        ("+Z", means.plus_z.reshape(1, 3)),
        ("-Z", means.minus_z.reshape(1, 3)),
    ]
    out = []
    for name, r in mats:
        a = calibrated_accel_mps2(r, bias, scale)[0]
        mag = float(np.linalg.norm(a))
        err = mag - G_MPS2
        out.append(
            {
                "pose": name,
                "calib_ax_mps2": float(a[0]),
                "calib_ay_mps2": float(a[1]),
                "calib_az_mps2": float(a[2]),
                "mag_mps2": mag,
                "err_vs_g_mps2": err,
            }
        )
    return pd.DataFrame(out)


def nominal_accel_scale_mps2_per_lsb() -> np.ndarray:
    """Номинальный масштаб (м/с²)/LSB при ±2g без six-axis."""
    s = G_MPS2 / ACCEL_LSB_PER_G
    return np.array([s, s, s], dtype=np.float64)


def build_json_output(
    *,
    mode: str,
    stationary_stats: Optional[Dict],
    six_pose_report: Optional[pd.DataFrame],
    verify_df: Optional[pd.DataFrame],
    accel_bias_raw: Optional[List[float]],
    accel_scale: Optional[List[float]],
    calibration_duration_s: float,
    fs_hz: float,
) -> dict:
    nominal_scale = nominal_accel_scale_mps2_per_lsb()

    gyro_bias_raw = None
    gyro_bias_rad_s = None
    accel_noise_std = None
    gyro_noise_std = None
    accel_cov = None
    gyro_cov = None

    if stationary_stats is not None:
        gyro_bias_raw = [
            float(stationary_stats["mean_gx_raw"]),
            float(stationary_stats["mean_gy_raw"]),
            float(stationary_stats["mean_gz_raw"]),
        ]
        gyro_bias_rad_s = stationary_stats["gyro_bias_rad_s"]
        accel_noise_std = stationary_stats["accel_noise_std_mps2"]
        gyro_noise_std = stationary_stats["gyro_noise_std_rad_s"]
        accel_cov = stationary_stats["accel_cov_mps2"]
        gyro_cov = stationary_stats["gyro_cov_rad_s2"]

    if accel_bias_raw is None:
        accel_bias_json = [0.0, 0.0, 0.0]
        accel_scale_json = nominal_scale.tolist()
        accel_bias_mps2 = [0.0, 0.0, 0.0]
    else:
        accel_bias_json = accel_bias_raw
        accel_scale_json = accel_scale
        b = np.array(accel_bias_raw, dtype=np.float64)
        s = np.array(accel_scale_json, dtype=np.float64)
        # Смещение в м/с² для альтернативной записи a = raw*s - b_si (см. README); основная — (raw-b)*s
        accel_bias_mps2 = (b * s).tolist()

    doc = {
        "sensor": "MPU6050",
        "accel_range": "+-2g",
        "gyro_range": "+-250dps",
        "accel_lsb_per_g": int(ACCEL_LSB_PER_G),
        "gyro_lsb_per_dps": int(GYRO_LSB_PER_DPS),
        "g": G_MPS2,
        "calibration_mode": mode,
        "accel_bias_raw": accel_bias_json,
        "accel_scale": accel_scale_json,
        "accel_bias_mps2": accel_bias_mps2,
        "accel_model_note": "Основная модель: a_mps2[i] = (raw[i] - accel_bias_raw[i]) * accel_scale[i]. "
        "Поле accel_bias_mps2 = bias_raw*scale — для формы a = raw*scale - accel_bias_mps2.",
        "gyro_bias_raw": gyro_bias_raw,
        "gyro_bias_rad_s": gyro_bias_rad_s,
        "accel_noise_std_mps2": accel_noise_std,
        "gyro_noise_std_rad_s": gyro_noise_std,
        "accel_noise_cov_mps2": accel_cov,
        "gyro_noise_cov_rad_s": gyro_cov,
        "sample_rate_hz_estimated": fs_hz,
        "calibration_duration_s": calibration_duration_s,
        "created_at": iso_now(),
    }

    if six_pose_report is not None:
        doc["six_pose_mean_raw"] = six_pose_report.to_dict(orient="records")
    if verify_df is not None:
        doc["six_pose_magnitude_check"] = verify_df.to_dict(orient="records")

    return doc


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Калибровка MPU6050 (Arduino + Python)")
    p.add_argument("--port", required=True, help="COM-порт, например COM5 или /dev/ttyUSB0")
    p.add_argument("--baud", type=int, default=115200)
    p.add_argument("--mode", choices=["stationary", "six_position"], required=True)
    p.add_argument("--duration", type=float, default=300.0, help="Длительность stationary, с")
    p.add_argument("--pose-duration", type=float, default=20.0, help="Длительность одной позы six_position, с")
    p.add_argument("--outdir", type=str, default="output", help="Каталог результатов")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_root = Path(args.outdir)
    plots_dir = out_root / "plots"
    ensure_dir(plots_dir)

    try:
        ser = serial.Serial(args.port, args.baud, timeout=0.3)
    except SerialException as e:
        print(f"Ошибка порта {args.port}: {e}", file=sys.stderr)
        return 2

    time.sleep(2.0)
    flush_serial(ser)

    raw_chunks: List[pd.DataFrame] = []
    stationary_stats = None
    report_stationary_df = None
    six_pose_report = None
    verify_df = None
    accel_bias_raw_list = None
    accel_scale_list = None
    total_duration = 0.0

    try:
        send_command(ser, "WHOAMI")

        if args.mode == "stationary":
            print("Режим stationary: неподвижно удерживайте модуль на столе.")
            input("Нажмите Enter, чтобы начать сбор...")
            df = collect_csv_stream(ser, duration_s=float(args.duration))
            raw_chunks.append(df)
            total_duration = float(args.duration)

            stationary_stats, report_stationary_df = stationary_calibration(df, total_duration)
            accel_bias_raw_list = [0.0, 0.0, 0.0]
            accel_scale_list = nominal_accel_scale_mps2_per_lsb().tolist()

            plot_timeseries(df, plots_dir, prefix="stationary_")
            gyro_rad = gyro_raw_to_rad_s(df[["gx_raw", "gy_raw", "gz_raw"]].to_numpy())
            gb = np.array(stationary_stats["gyro_bias_rad_s"], dtype=np.float64)
            gyro_c = gyro_rad - gb.reshape(1, 3)
            accel_m = accel_raw_to_mps2(df[["ax_raw", "ay_raw", "az_raw"]].to_numpy())
            am = np.mean(accel_m, axis=0)
            accel_c = accel_m - am.reshape(1, 3)
            plot_histograms(accel_c, gyro_c, plots_dir, prefix="stationary_")

        elif args.mode == "six_position":
            print("Режим six_position: понадобится поочерёдно поставить модуль шестью гранями.")
            sm, six_pose_report = six_position_calibration(
                ser, pose_duration_s=float(args.pose_duration), raw_chunks=raw_chunks
            )
            bias, scale = accel_bias_scale_from_six_poses(sm)
            accel_bias_raw_list = bias.tolist()
            accel_scale_list = scale.tolist()
            verify_df = verify_six_poses(sm, bias, scale)
            total_duration = float(args.pose_duration) * 6.0

            df_all = pd.concat(raw_chunks, ignore_index=True)
            plot_timeseries(df_all, plots_dir, prefix="sixpos_")

            # Шум акселя по шестипозиционным данным не интерпретируем здесь как один режим — пропускаем гистограммы
            # или строим по последнему фрагменту (не требуется ТЗ). Оставим только временные ряды.

        else:
            raise AssertionError("Неизвестный режим")

        df_log = pd.concat(raw_chunks, ignore_index=True) if raw_chunks else pd.DataFrame()
        raw_csv_path = out_root / "raw_log.csv"
        df_log.to_csv(raw_csv_path, index=False)

        fs_hz = float(estimate_sample_rate_hz(df_log["time_ms"].to_numpy())) if len(df_log) else float("nan")

        json_doc = build_json_output(
            mode=args.mode,
            stationary_stats=stationary_stats,
            six_pose_report=six_pose_report,
            verify_df=verify_df,
            accel_bias_raw=accel_bias_raw_list,
            accel_scale=accel_scale_list,
            calibration_duration_s=total_duration,
            fs_hz=fs_hz,
        )
        write_json(out_root / "imu_calibration.json", json_doc)

        # calibration_report.csv — конкатенация таблиц
        frames = []
        if report_stationary_df is not None:
            frames.append(report_stationary_df.assign(section="stationary"))
        if six_pose_report is not None:
            frames.append(six_pose_report.assign(section="six_pose_means"))
        if verify_df is not None:
            frames.append(verify_df.assign(section="six_pose_verify"))

        if frames:
            pd.concat(frames, ignore_index=True).to_csv(out_root / "calibration_report.csv", index=False)
        else:
            pd.DataFrame({"message": ["no report rows"]}).to_csv(out_root / "calibration_report.csv", index=False)

        print(f"\nГотово. Файлы в каталоге: {out_root.resolve()}")
        print(f"  raw_log.csv, calibration_report.csv, imu_calibration.json, plots/")

    except (TimeoutError, ConnectionError, ValueError) as e:
        print(f"Ошибка выполнения: {e}", file=sys.stderr)
        send_command(ser, "STOP_STREAM")
        return 1
    finally:
        try:
            flush_serial(ser)
            ser.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
