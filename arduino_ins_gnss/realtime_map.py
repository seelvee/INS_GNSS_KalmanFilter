# -*- coding: utf-8 -*-
"""
Realtime карта (lat/lon) по данным с Arduino:
- GNSS: точки из lat/lon.
- INS: интеграция ускорения в ENU на лету с привязкой старта к первой валидной GNSS-точке.

Поддерживаемые форматы CSV-строк:
1) physical: ... ax,ay,az,gx,gy,gz[,roll_deg,pitch_deg,yaw_deg]
2) dmp_raw: ... ax_raw,ay_raw,az_raw,gx_raw,gy_raw,gz_raw[,roll_dmp_deg,pitch_dmp_deg,yaw_dmp_deg]

Запуск:
  python realtime_map.py
"""
import math
import threading
from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import serial
import serial.tools.list_ports

BAUD = 115200
GRAVITY = 9.80665

# DMP raw assumptions for MPU6050 defaults (same as analyzer).
ACC_SCALE_RAW = 9.80665 / 16384.0
GYRO_SCALE_RAW = (math.pi / 180.0) / 131.0

_lock = threading.Lock()
state = {
    "header_idx": None,
    "is_raw": False,
    "latest": None,
    "has_data": False,
}


def parse_float(s: str) -> float:
    s = (s or "").strip().lower()
    if s in ("", "nan", "nan(ind)"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> np.ndarray:
    # Local tangent approximation.
    d_lat = math.radians(lat - lat0)
    d_lon = math.radians(lon - lon0)
    r = 6378137.0
    n = d_lat * r
    e = d_lon * r * math.cos(math.radians(lat0))
    return np.array([e, n, 0.0], dtype=float)


def enu_to_ll(e: float, n: float, lat0: float, lon0: float) -> tuple[float, float]:
    r = 6378137.0
    lat = lat0 + math.degrees(n / r)
    lon = lon0 + math.degrees(e / (r * math.cos(math.radians(lat0))))
    return lat, lon


def euler_to_c_nb(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    # Body (X fwd, Y right, Z down) -> ENU.
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def read_thread(ser: serial.Serial):
    while getattr(ser, "_running", True) and ser.is_open:
        try:
            line = ser.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith("#"):
                continue

            if text.startswith("time_ms,"):
                header = [h.strip().lower() for h in text.split(",")]
                idx = {h: i for i, h in enumerate(header)}
                with _lock:
                    state["header_idx"] = idx
                    state["is_raw"] = ("ax_raw" in idx and "gz_raw" in idx)
                continue

            with _lock:
                idx = state["header_idx"]
            if idx is None:
                continue

            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 13:
                continue

            def v(name: str) -> float:
                i = idx.get(name)
                if i is None or i >= len(parts):
                    return float("nan")
                return parse_float(parts[i])

            is_raw = ("ax_raw" in idx and "gz_raw" in idx)
            sample = {
                "time_ms": int(v("time_ms")) if not math.isnan(v("time_ms")) else 0,
                "lat": v("lat"),
                "lon": v("lon"),
                "speed_mps": v("speed_mps"),
                "ax": v("ax_raw") if is_raw else v("ax"),
                "ay": v("ay_raw") if is_raw else v("ay"),
                "az": v("az_raw") if is_raw else v("az"),
                "gx": v("gx_raw") if is_raw else v("gx"),
                "gy": v("gy_raw") if is_raw else v("gy"),
                "gz": v("gz_raw") if is_raw else v("gz"),
                "roll": v("roll_deg") if not math.isnan(v("roll_deg")) else v("roll_dmp_deg"),
                "pitch": v("pitch_deg") if not math.isnan(v("pitch_deg")) else v("pitch_dmp_deg"),
                "yaw": v("yaw_deg") if not math.isnan(v("yaw_deg")) else v("yaw_dmp_deg"),
                "is_raw": is_raw,
            }
            with _lock:
                state["latest"] = sample
                state["has_data"] = True
        except Exception:
            pass


def main():
    ports = list(serial.tools.list_ports.comports())
    port_guess: Optional[str] = None
    for p in ports:
        if "Arduino" in (p.description or "") or "CH340" in (p.description or "") or "USB" in (p.description or ""):
            port_guess = p.device
            break
    if not port_guess and ports:
        port_guess = ports[0].device
    if not port_guess:
        print("COM-порт не найден.")
        return

    port = input(f"Порт (Enter = {port_guess}): ").strip() or port_guess
    try:
        ser = serial.Serial(port, BAUD, timeout=0.05)
    except Exception as e:
        print("Ошибка открытия порта:", e)
        return

    ser._running = True
    threading.Thread(target=read_thread, args=(ser,), daemon=True).start()

    fig, ax = plt.subplots(figsize=(9, 7))
    gnss_line, = ax.plot([], [], "g.-", markersize=2, linewidth=1, label="GNSS")
    ins_line, = ax.plot([], [], "r-", linewidth=1.5, label="INS (realtime)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True)
    ax.legend()

    gnss_lat, gnss_lon = [], []
    ins_lat, ins_lon = [], []
    lat0 = lon0 = None
    p_enu = np.zeros(3)
    v_enu = np.zeros(3)
    last_t = None

    def update(_frame):
        nonlocal lat0, lon0, p_enu, v_enu, last_t
        with _lock:
            sample = state["latest"]
            ok = state["has_data"]
        if not ok or sample is None:
            return gnss_line, ins_line

        t_ms = sample["time_ms"]
        lat = sample["lat"]
        lon = sample["lon"]
        roll = sample["roll"]
        pitch = sample["pitch"]
        yaw = sample["yaw"]
        ax_b, ay_b, az_b = sample["ax"], sample["ay"], sample["az"]
        is_raw = sample["is_raw"]

        if lat0 is None and not (math.isnan(lat) or math.isnan(lon)):
            lat0, lon0 = lat, lon
            gnss_lat.append(lat)
            gnss_lon.append(lon)
            ins_lat.append(lat)
            ins_lon.append(lon)
            last_t = t_ms
            return gnss_line, ins_line

        if lat0 is None:
            return gnss_line, ins_line

        # GNSS trace.
        if not (math.isnan(lat) or math.isnan(lon)):
            gnss_lat.append(lat)
            gnss_lon.append(lon)

        # INS propagation.
        if last_t is None:
            last_t = t_ms
        dt = max(1e-3, min(1.0, (t_ms - last_t) / 1000.0))
        last_t = t_ms

        if is_raw:
            ax_b *= ACC_SCALE_RAW
            ay_b *= ACC_SCALE_RAW
            az_b *= ACC_SCALE_RAW

        if not any(math.isnan(x) for x in (ax_b, ay_b, az_b, roll, pitch, yaw)):
            c_nb = euler_to_c_nb(roll, pitch, yaw)
            f_nav = c_nb @ np.array([ax_b, ay_b, az_b], dtype=float)
            a_nav = f_nav + np.array([0.0, 0.0, -GRAVITY], dtype=float)
            v_enu = v_enu + a_nav * dt
            p_enu = p_enu + v_enu * dt + 0.5 * a_nav * dt * dt

            lat_i, lon_i = enu_to_ll(p_enu[0], p_enu[1], lat0, lon0)
            ins_lat.append(lat_i)
            ins_lon.append(lon_i)

        # Soft GNSS anchoring to reduce unbounded drift.
        if not (math.isnan(lat) or math.isnan(lon)):
            p_gnss = ll_to_enu(lat, lon, lat0, lon0)
            p_enu = 0.9 * p_enu + 0.1 * p_gnss

        if gnss_lat:
            gnss_line.set_data(gnss_lon, gnss_lat)
        if ins_lat:
            ins_line.set_data(ins_lon, ins_lat)
        if len(gnss_lat) + len(ins_lat) > 5:
            all_lat = np.array(gnss_lat + ins_lat, dtype=float)
            all_lon = np.array(gnss_lon + ins_lon, dtype=float)
            lat_min, lat_max = np.nanmin(all_lat), np.nanmax(all_lat)
            lon_min, lon_max = np.nanmin(all_lon), np.nanmax(all_lon)
            dlat = max(1e-5, (lat_max - lat_min) * 0.15)
            dlon = max(1e-5, (lon_max - lon_min) * 0.15)
            ax.set_ylim(lat_min - dlat, lat_max + dlat)
            ax.set_xlim(lon_min - dlon, lon_max + dlon)
        return gnss_line, ins_line

    anim = FuncAnimation(fig, update, interval=100, blit=False, cache_frame_data=False)
    print("Realtime карта запущена. Закройте окно для выхода.")
    plt.tight_layout()
    plt.show()
    ser._running = False
    ser.close()


if __name__ == "__main__":
    main()

