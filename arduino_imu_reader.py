# -*- coding: utf-8 -*-
"""
Чтение с Arduino только чистых (сырых) значений акселерометра и гироскопа.
Формат строки с платы: ax,ay,az,gx,gy,gz (целые числа, через запятую).
Обработка (ориентация, перевод в навигационную СК) — в imu_body_to_nav и фильтре Калмана.
"""

import time
import re
from typing import Optional, Callable, Tuple

import numpy as np
from imu_body_to_nav import (
    ArduinoIMUScales,
    BodyToNavProcessor,
    raw_to_physical,
)
from ins_gnss_filter import INSGNSSKalmanFilter


def parse_arduino_imu_line(line: str) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Разбор одной строки от Arduino: ax,ay,az,gx,gy,gz (чистые значения в LSB).

    Returns
    -------
    (raw_accel, raw_gyro) или None при ошибке разбора.
    raw_accel, raw_gyro : (3,) int/float
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = re.findall(r"-?\d+", line)
    if len(parts) >= 6:
        raw_accel = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        raw_gyro = np.array([float(parts[3]), float(parts[4]), float(parts[5])])
        return raw_accel, raw_gyro
    return None


def read_arduino_imu_loop(
    serial_port: "serial.Serial",
    dt: float = 0.01,
    scales: Optional[ArduinoIMUScales] = None,
    on_nav_accel: Optional[
        Callable[[np.ndarray, float, float, float, np.ndarray], None]
    ] = None,
    on_raw: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
    stop_event: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Цикл чтения с Arduino: получаем только сырые ax,ay,az,gx,gy,gz;
    внутри переводим в навигационную СК и при необходимости вызываем callback.

    Parameters
    ----------
    serial_port : открытый serial.Serial (pyserial)
    dt : шаг по времени, с (или оценивается по таймстампам)
    scales : масштабы для сырых значений; None — по умолчанию MPU6050
    on_nav_accel : callback(accel_nav, roll, pitch, yaw, gyro_body) после перевода в nav.
                   gyro_body — (3,) рад/с в связанной СК.
    on_raw : callback(raw_accel, raw_gyro) — только сырые данные (LSB)
    stop_event : функция без аргументов, возвращает True для выхода из цикла
    """
    if scales is None:
        scales = ArduinoIMUScales()
    processor = BodyToNavProcessor(yaw_fixed=0.0)
    t_prev = time.perf_counter()
    while True:
        if stop_event and stop_event():
            break
        line = serial_port.readline()
        if not line:
            continue
        try:
            decoded = line.decode("utf-8", errors="ignore")
        except Exception:
            continue
        parsed = parse_arduino_imu_line(decoded)
        if parsed is None:
            continue
        raw_accel, raw_gyro = parsed
        t_cur = time.perf_counter()
        dt_actual = min(t_cur - t_prev, 0.5)
        t_prev = t_cur
        if on_raw:
            on_raw(raw_accel, raw_gyro)
        accel_body, gyro_body = raw_to_physical(raw_accel, raw_gyro, scales)
        processor.update_orientation(gyro_body[0], gyro_body[1], gyro_body[2], dt_actual)
        accel_nav = processor.accel_body_to_nav(accel_body)
        roll, pitch, yaw = processor.get_orientation()
        if on_nav_accel:
            on_nav_accel(accel_nav, roll, pitch, yaw, gyro_body)


def run_with_kalman_from_arduino(
    serial_port: "serial.Serial",
    kf: INSGNSSKalmanFilter,
    dt: float = 0.01,
    scales: Optional[ArduinoIMUScales] = None,
    gnss_callback: Optional[Callable[[], Optional[Tuple[np.ndarray, Optional[np.ndarray]]]]] = None,
    stop_event: Optional[Callable[[], bool]] = None,
) -> None:
    """
    Чтение сырых данных с Arduino, перевод в навигационную СК, подача в фильтр Калмана.
    ГНСС (если есть): gnss_callback() возвращает (pos_enu, vel_enu) или (pos_enu, None) или None.
    """
    scales = scales or ArduinoIMUScales()

    def on_nav(accel_nav: np.ndarray, roll: float, pitch: float, yaw: float, gyro_body: np.ndarray):
        kf.predict(accel_nav)
        if gnss_callback:
            meas = gnss_callback()
            if meas is not None:
                pos, vel = meas
                if vel is not None:
                    kf.update(np.concatenate([np.asarray(pos).ravel()[:3], np.asarray(vel).ravel()[:3]]))
                else:
                    kf.update(np.asarray(pos).ravel()[:3])

    read_arduino_imu_loop(
        serial_port,
        dt=dt,
        scales=scales,
        on_nav_accel=on_nav,
        on_raw=None,
        stop_event=stop_event,
    )


# Пример использования без платы (тест разбора строки)
if __name__ == "__main__":
    # Тест парсера
    line = "1234,-567,8901,100,-200,50"
    r = parse_arduino_imu_line(line)
    print("Парсер:", r)
    if r:
        raw_accel, raw_gyro = r
        print("Сырые акселерометр:", raw_accel, "гироскоп:", raw_gyro)
