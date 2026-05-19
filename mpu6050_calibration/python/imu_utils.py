"""
Утилиты для калибровки MPU6050: перевод LSB → физические единицы, парсинг CSV,
ковариационные матрицы шума, сохранение графиков.

Масштабы (как в ТЗ):
  аксель ±2g:   16384 LSB/g
  гиро ±250 °/с: 131 LSB/(°/с)
  g = 9.80665 м/с²
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


G_MPS2 = 9.80665
ACCEL_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0


def accel_raw_to_mps2(raw: np.ndarray) -> np.ndarray:
    """Перевод сырых отсчётов акселерометра в м/с² (номинальный шкала ±2g)."""
    return (raw.astype(np.float64) / ACCEL_LSB_PER_G) * G_MPS2


def gyro_raw_to_rad_s(raw: np.ndarray) -> np.ndarray:
    """Перевод сырых отсчётов гироскопа в рад/с (номинальный шкала ±250 °/с)."""
    dps = raw.astype(np.float64) / GYRO_LSB_PER_DPS
    return np.deg2rad(dps)


def parse_csv_line(line: str) -> Optional[Tuple[int, np.ndarray]]:
    """
    Разбор одной строки CSV потока:
    time_ms,ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw,temp_raw

    Возвращает (time_ms, vector_int16[7]) или None при ошибке.
    """
    s = line.strip()
    if not s or s.startswith("#"):
        return None
    parts = s.split(",")
    if len(parts) != 8:
        return None
    try:
        t_ms = int(parts[0])
        vals = np.array([int(parts[i]) for i in range(1, 8)], dtype=np.int64)
    except ValueError:
        return None
    return t_ms, vals


def load_raw_csv(path: Path) -> pd.DataFrame:
    """Читает сохранённый raw_log.csv."""
    df = pd.read_csv(path)
    expected = ["time_ms", "ax_raw", "ay_raw", "az_raw", "gx_raw", "gy_raw", "gz_raw", "temp_raw"]
    for c in expected:
        if c not in df.columns:
            raise ValueError(f"В файле нет колонки '{c}'")
    return df


def estimate_sample_rate_hz(time_ms: np.ndarray) -> float:
    """Оценка средней частоты по меткам времени (Гц)."""
    if time_ms.size < 2:
        return float("nan")
    dt = np.diff(time_ms.astype(np.float64)) / 1000.0
    dt = dt[dt > 0]
    if dt.size == 0:
        return float("nan")
    return float(1.0 / np.median(dt))


def covariance_matrix(samples_centered: np.ndarray) -> np.ndarray:
    """
    Выборочная ковариация (несмещённая): samples_centered shape (N, 3).
    """
    if samples_centered.shape[0] < 2:
        raise ValueError("Недостаточно строк для ковариации")
    return np.cov(samples_centered, rowvar=False, bias=False)


def noise_std_per_axis(samples_centered: np.ndarray) -> np.ndarray:
    """Стандартное отклонение по каждому столбцу (оси)."""
    return np.std(samples_centered, axis=0, ddof=1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_timeseries(df: pd.DataFrame, out_dir: Path, prefix: str = "") -> None:
    """Графики ax, ay, az и gx, gy, gz во времени (сырые коды)."""
    ensure_dir(out_dir)
    t = df["time_ms"].to_numpy(dtype=float) / 1000.0

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, name in enumerate(["ax_raw", "ay_raw", "az_raw"]):
        axs[i].plot(t, df[name].to_numpy(), lw=0.6)
        axs[i].set_ylabel(name)
    axs[-1].set_xlabel("t, s")
    fig.suptitle("Акселерометр (сырые LSB)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}accel_time.png", dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, name in enumerate(["gx_raw", "gy_raw", "gz_raw"]):
        axs[i].plot(t, df[name].to_numpy(), lw=0.6)
        axs[i].set_ylabel(name)
    axs[-1].set_xlabel("t, s")
    fig.suptitle("Гироскоп (сырые LSB)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}gyro_time.png", dpi=150)
    plt.close(fig)


def plot_histograms(
    accel_noise_mps2: np.ndarray,
    gyro_noise_rad_s: np.ndarray,
    out_dir: Path,
    prefix: str = "",
) -> None:
    """Гистограммы центрированного шума акселя (м/с²) и гиро (рад/с)."""
    ensure_dir(out_dir)
    labels_a = ["ax", "ay", "az"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for i in range(3):
        axs[i].hist(accel_noise_mps2[:, i], bins=40, density=True, alpha=0.85)
        axs[i].set_title(labels_a[i])
        axs[i].set_xlabel("м/с²")
    fig.suptitle("Шум акселерометра (после вычитания среднего)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}accel_noise_hist.png", dpi=150)
    plt.close(fig)

    labels_g = ["gx", "gy", "gz"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    for i in range(3):
        axs[i].hist(gyro_noise_rad_s[:, i], bins=40, density=True, alpha=0.85)
        axs[i].set_title(labels_g[i])
        axs[i].set_xlabel("рад/с")
    fig.suptitle("Шум гироскопа (после вычитания bias)")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}gyro_noise_hist.png", dpi=150)
    plt.close(fig)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


@dataclass
class SixPoseMeans:
    """Средние сырые ускорения для шести ориентаций."""

    plus_x: np.ndarray
    minus_x: np.ndarray
    plus_y: np.ndarray
    minus_y: np.ndarray
    plus_z: np.ndarray
    minus_z: np.ndarray


def accel_bias_scale_from_six_poses(means: SixPoseMeans, g_mps2: float = G_MPS2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Простая калибровка по ТЗ (сырые коды по осям):

      bias_axis = (mean_plus + mean_minus) / 2
      scale_axis = 2*g / (mean_plus - mean_minus)   [м/с² за один LSB по формуле ниже]

    Ускорение в м/с²: a = (raw - bias) * scale
    где scale имеет размерность (м/с²)/LSB.

    Проверка: для оси X при raw = mean_plus: (mean_plus - bias) = Δ/2,
    a = Δ/2 * (2g/Δ) = g.
    """
    bx = (means.plus_x[0] + means.minus_x[0]) * 0.5
    by = (means.plus_y[1] + means.minus_y[1]) * 0.5
    bz = (means.plus_z[2] + means.minus_z[2]) * 0.5
    bias = np.array([bx, by, bz], dtype=np.float64)

    dx = float(means.plus_x[0] - means.minus_x[0])
    dy = float(means.plus_y[1] - means.minus_y[1])
    dz = float(means.plus_z[2] - means.minus_z[2])
    if dx == 0 or dy == 0 or dz == 0:
        raise ValueError("Нулевой размах +/- для одной из осей — проверьте положения датчика")

    sx = (2.0 * g_mps2) / dx
    sy = (2.0 * g_mps2) / dy
    sz = (2.0 * g_mps2) / dz
    scale = np.array([sx, sy, sz], dtype=np.float64)
    return bias, scale


def calibrated_accel_mps2(raw_xyz: np.ndarray, bias: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """raw_xyz (N,3) int/float → ускорение м/с² по осям датчика."""
    r = raw_xyz.astype(np.float64)
    return (r - bias.reshape(1, 3)) * scale.reshape(1, 3)


def magnitude_mps2(a_mps2: np.ndarray) -> np.ndarray:
    return np.linalg.norm(a_mps2, axis=1)
