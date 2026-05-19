# -*- coding: utf-8 -*-
"""
Инерциалка: ориентация по уравнению Пуассона, ускорение из удельной силы и гравитации.

Уравнение Пуассона (кинематика вращения):
  dC/dt = C * [ω×]_b,
где C = C_nb — матрица поворота из связанной СК в навигационную (ENU),
ω — угловая скорость в связанной СК (рад/с) с гироскопа, [ω×] — кососимметрическая матрица.

Интегрирование за шаг dt: C_{k+1} = C_k @ R_step, R_step = exp([ω×]*dt) — по формуле Родрига.

Ускорение в нав. СК: a_nav = C @ f_body + g_nav,
где f_body — удельная сила (показания акселерометра в body), g_nav = (0, 0, -g) в ENU.

Body: X вперёд, Y вправо, Z вниз. Nav: East, North, Up.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

# м/с²
GRAVITY = 9.80665

# Смещение нуля гироскопа (калибровка), град/с.
# MPU6050: ±250°/s => 131 LSB/(град/с); ±500°/s => 65.5 LSB/(град/с). Match firmware.
GYRO_BIAS_DEG_PER_SEC = (-5.362898, 0.597485, -0.228809)
GYRO_LSB_PER_DEG_PER_SEC = 65.5   # ±500 deg/s (MPU6050_RANGE_500_DEG)
GYRO_BIAS_LSB = tuple(b * GYRO_LSB_PER_DEG_PER_SEC for b in GYRO_BIAS_DEG_PER_SEC)


@dataclass
class ArduinoIMUScales:
    """
    Только для скетчей, которые шлют сырые LSB (например arduino_imu_raw.ino).
    arduino_ins_gnss.ino отдаёт уже м/с² и град/с — там эти масштабы не использовать.
    """
    accel_scale: float = 9.80665 / 8192.0   # ±4g => 8192 LSB/g (match MPU6050_RANGE_4_G)
    gyro_scale: float = (np.pi / 180.0) / 65.5   # ±500 deg/s => 65.5 LSB/(deg/s)
    accel_bias: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias: Tuple[float, float, float] = GYRO_BIAS_LSB


def raw_to_physical(
    raw_accel: np.ndarray,
    raw_gyro: np.ndarray,
    scales: ArduinoIMUScales,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Перевод сырых значений с Arduino в физические величины.

    Parameters
    ----------
    raw_accel : (3,) или (ax, ay, az) в LSB (чистые значения с датчика)
    raw_gyro : (3,) или (gx, gy, gz) в LSB
    scales : масштабы и смещения

    Returns
    -------
    accel_body : (3,) м/с² — удельная сила в связанной СК (X вперёд, Y вправо, Z вниз)
    gyro_body : (3,) рад/с — угловая скорость в связанной СК
    """
    raw_accel = np.asarray(raw_accel, dtype=float).ravel()[:3]
    raw_gyro = np.asarray(raw_gyro, dtype=float).ravel()[:3]
    ab = np.array(scales.accel_bias)
    gb = np.array(scales.gyro_bias)
    accel_body = (raw_accel - ab) * scales.accel_scale
    gyro_body = (raw_gyro - gb) * scales.gyro_scale
    return accel_body, gyro_body


def skew_omega(wx: float, wy: float, wz: float) -> np.ndarray:
    """Матрица [ω×] (кососимметрическая) для угловой скорости в связанной СК."""
    return np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ])


def rodrigues_rotation(omega_body: np.ndarray, dt: float) -> np.ndarray:
    """
    Матрица поворота за шаг dt при постоянной угловой скорости ω в связанной СК.
    R = exp([ω×] * dt) по формуле Родрига; при малом θ используется разложение.
    """
    wx, wy, wz = float(omega_body[0]), float(omega_body[1]), float(omega_body[2])
    theta = np.sqrt(wx * wx + wy * wy + wz * wz) * dt
    if theta < 1e-12:
        # Первый порядок: R ≈ I + [ω×]*dt
        Om = skew_omega(wx, wy, wz)
        return np.eye(3) + Om * dt
    # Родрига: exp([ω×]*dt) = I + (sin θ/θ)*[ω×]*dt + ((1-cos θ)/θ²)*([ω×]*dt)²
    Om = skew_omega(wx, wy, wz)
    Omd = Om * dt
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    R = np.eye(3) + (sin_th / theta) * Omd + ((1.0 - cos_th) / (theta * theta)) * (Omd @ Omd)
    return R


def orthonormalize_rotation(C: np.ndarray) -> np.ndarray:
    """Восстановление ортогональности матрицы поворота (дрейф из-за интегрирования)."""
    U, _, Vt = np.linalg.svd(C)
    return U @ Vt


def rotation_body_to_nav_from_yaw(yaw: float) -> np.ndarray:
    """
    Матрица C_nb из одного курса (крен и тангаж = 0).
    Body: X вперёд (North при yaw=0), Y вправо (East), Z вниз.
    Nav: East, North, Up.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    # Столбцы = оси body в координатах nav: X_body = (-sin(yaw), cos(yaw), 0), Y_body = (cos(yaw), sin(yaw), 0), Z_body = (0, 0, -1)
    return np.array([
        [-s, c, 0.0],
        [c, s, 0.0],
        [0.0, 0.0, -1.0],
    ])


def rotation_body_to_nav(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Матрица перехода из связанной СК в навигационную (ENU).
    Body: X вперёд (North при yaw=0), Y вправо (East), Z вниз.
    Nav: East, North, Up.
    C_nb: v_nav = C_nb @ v_body.
    """
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rotation_to_euler(C: np.ndarray) -> Tuple[float, float, float]:
    """Извлечение углов Эйлера (roll, pitch, yaw) из матрицы C_nb. Порядок ZYX."""
    roll = np.arctan2(C[2, 1], C[2, 2])
    pitch = np.arcsin(np.clip(-C[2, 0], -1.0, 1.0))
    yaw = np.arctan2(C[1, 0], C[0, 0])
    return roll, pitch, yaw


class BodyToNavProcessor:
    """
    Ориентация по уравнению Пуассона: dC/dt = C * [ω×]_b.
    Хранится матрица поворота C_nb (body → nav); на каждом шаге
    C_new = C_old @ R_step, где R_step = exp([ω×]*dt) по Родрига.
    Ускорение в нав. СК: a_nav = C @ f_body + g_nav (удельная сила и гравитация).
    """

    def __init__(self, initial_yaw: float = 0.0, yaw_fixed: Optional[float] = None):
        if yaw_fixed is not None:
            initial_yaw = yaw_fixed
        yaw = float(initial_yaw)
        self._C = rotation_body_to_nav_from_yaw(yaw)

    def update_orientation(self, wx: float, wy: float, wz: float, dt: float) -> Tuple[float, float, float]:
        """
        Уравнение Пуассона: приращение ориентации по угловой скорости в связанной СК.
        C_{k+1} = C_k @ exp([ω×]*dt); exp через формулу Родрига.
        """
        dt = max(1e-9, min(float(dt), 1.0))
        omega = np.array([wx, wy, wz], dtype=float)
        R_step = rodrigues_rotation(omega, dt)
        self._C = self._C @ R_step
        self._C = orthonormalize_rotation(self._C)
        return self.get_orientation()

    def accel_body_to_nav(self, accel_body: np.ndarray) -> np.ndarray:
        """a_nav = C @ f_body + g_nav. Возврат (3,) в ENU: [East, North, Up] м/с²."""
        accel_body = np.asarray(accel_body, dtype=float).ravel()[:3]
        f_nav = self._C @ accel_body
        g_nav = np.array([0.0, 0.0, -GRAVITY])
        return f_nav + g_nav

    def get_orientation(self) -> Tuple[float, float, float]:
        roll, pitch, yaw = rotation_to_euler(self._C)
        yaw = (yaw + np.pi) % (2.0 * np.pi) - np.pi
        return roll, pitch, yaw


def raw_arduino_to_nav_accel(
    raw_accel: np.ndarray,
    raw_gyro: np.ndarray,
    processor: BodyToNavProcessor,
    dt: float,
    scales: Optional[ArduinoIMUScales] = None,
) -> np.ndarray:
    """
    Полный цикл: сырые значения с Arduino → ориентация по гироскопу → ускорение в навигационной СК.

    Parameters
    ----------
    raw_accel, raw_gyro : (3,) сырые значения (LSB)
    processor : экземпляр BodyToNavProcessor (хранит состояние ориентации)
    dt : шаг по времени, с
    scales : если None, используется ArduinoIMUScales()

    Returns
    -------
    accel_nav : (3,) м/с² в НЭЗ (East, North, Up)
    """
    if scales is None:
        scales = ArduinoIMUScales()
    accel_body, gyro_body = raw_to_physical(raw_accel, raw_gyro, scales)
    processor.update_orientation(gyro_body[0], gyro_body[1], gyro_body[2], dt)
    return processor.accel_body_to_nav(accel_body)
