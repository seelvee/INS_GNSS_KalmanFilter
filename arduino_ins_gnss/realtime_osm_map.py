# -*- coding: utf-8 -*-
"""
Realtime карта OpenStreetMap (Leaflet) с GNSS и INS треками.

Запуск с Arduino (COM):
  python realtime_osm_map.py
  python realtime_osm_map.py --port COM3

Воспроизведение записанного CSV (тот же парсер, та же карта в браузере):
  python realtime_osm_map.py --file ins_gnss_20260430_200111.csv
  python realtime_osm_map.py --file ../ins_gnss_20260303_221628.csv --replay-speed 25

Зависимости:
  pip install flask pyserial numpy
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time
import webbrowser
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import serial
import serial.tools.list_ports
from flask import Flask, jsonify

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
try:
    from ins_gnss_filter import FilterConfig, INSGNSSKalmanFilter
except Exception:
    @dataclass
    class FilterConfig:
        dt: float
        sigma_acc: float
        sigma_gnss_pos: float
        sigma_gnss_vel: float
        use_gnss_velocity: bool = False

    class INSGNSSKalmanFilter:
        """Fallback Kalman ENU: x = [E, N, U, vE, vN, vU]."""

        def __init__(self, config: FilterConfig, initial_state: np.ndarray):
            self.config = config
            self.x = np.asarray(initial_state, dtype=float).reshape(6)
            self.P = np.eye(6) * 10.0
            self._set_dt(float(config.dt))
            self.R = np.eye(3) * (config.sigma_gnss_pos ** 2)

        def _set_dt(self, dt: float) -> None:
            dt = float(np.clip(dt, 1e-3, 1.0))
            self.F = np.array([
                [1.0, 0.0, 0.0, dt, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, dt, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, dt],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ], dtype=float)
            self.B = np.array([
                [0.5 * dt * dt, 0.0, 0.0],
                [0.0, 0.5 * dt * dt, 0.0],
                [0.0, 0.0, 0.5 * dt * dt],
                [dt, 0.0, 0.0],
                [0.0, dt, 0.0],
                [0.0, 0.0, dt],
            ], dtype=float)
            q = self.config.sigma_acc ** 2
            G = np.vstack([self.B[:3], self.B[3:] * 0.5])
            self.Q = (G * q) @ G.T + np.eye(6) * 1e-9
            self.H = np.array([
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ], dtype=float)

        def predict(self, acceleration: np.ndarray, dt: float | None = None) -> np.ndarray:
            step = self.config.dt if dt is None else float(np.clip(dt, 1e-3, 1.0))
            self._set_dt(step)
            a = np.asarray(acceleration, dtype=float).reshape(3)
            # p = p + v*dt + 0.5*a*dt^2, v = v + a*dt
            self.x = self.F @ self.x + self.B @ a
            self.P = self.F @ self.P @ self.F.T + self.Q
            return self.x.copy()

        def update(self, z_gnss: np.ndarray) -> np.ndarray:
            z = np.asarray(z_gnss, dtype=float).reshape(3)
            y = z - self.H @ self.x
            S = self.H @ self.P @ self.H.T + self.R
            K = self.P @ self.H.T @ np.linalg.solve(S, np.eye(3))
            self.x = self.x + K @ y
            I = np.eye(6)
            self.P = (I - K @ self.H) @ self.P
            return self.x.copy()

        def get_position(self) -> np.ndarray:
            return self.x[:3].copy()

BAUD = 115200
GRAVITY = 9.80665
ACC_LSB_PER_G = 16384.0
GYRO_LSB_PER_DPS = 131.0
# MPU6050 scales for Arduino config: FS_2G and FS_250DPS.
ACC_SCALE_RAW = GRAVITY / ACC_LSB_PER_G
GYRO_SCALE_RAW = np.deg2rad(1.0 / GYRO_LSB_PER_DPS)

app = Flask(__name__)

_lock = threading.Lock()
state = {
    "header_idx": None,
    "latest": None,
    "has_data": False,
    "gnss_track": [],
    "ins_track": [],
    "kalman_track": [],
    "lat0": None,
    "lon0": None,
    "p_enu": np.zeros(3),
    "v_enu": np.zeros(3),
    "last_t_ms": None,
    "kf": None,
    "mode": "serial",
    "replay_path": None,
    "replay_finished": False,
    "arduino_header": None,
    "arduino_rows": [],
    "last_debug": {
        "dt": 0.0,
        "acc_norm": float("nan"),
        "acc_minus_g": float("nan"),
        "a_n": float("nan"),
        "a_e": float("nan"),
        "v_n_ins": 0.0,
        "v_e_ins": 0.0,
        "dmp_ok": 1,
        "fix": float("nan"),
        "sats": float("nan"),
    },
    "skipped_imu": 0,
    "parse_errors": 0,
}


def parse_float(s: str) -> float:
    s = (s or "").strip().lower()
    if s in ("", "nan", "nan(ind)"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def enu_to_ll(e: float, n: float, lat0: float, lon0: float):
    r = 6378137.0
    lat = lat0 + math.degrees(n / r)
    lon = lon0 + math.degrees(e / (r * math.cos(math.radians(lat0))))
    return lat, lon


def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float):
    r = 6378137.0
    d_lat = math.radians(lat - lat0)
    d_lon = math.radians(lon - lon0)
    n = d_lat * r
    e = d_lon * r * math.cos(math.radians(lat0))
    return np.array([e, n, 0.0], dtype=float)


def euler_to_c_nb(roll_deg: float, pitch_deg: float, yaw_deg: float):
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def build_sample_from_parts(parts: list[str], idx: dict[str, int]) -> dict | None:
    if len(parts) < 13:
        return None

    def v(name: str):
        i = idx.get(name)
        if i is None or i >= len(parts):
            return float("nan")
        return parse_float(parts[i])

    accel_is_raw = "ax_raw" in idx and "ay_raw" in idx and "az_raw" in idx
    gyro_is_raw = "gx_raw" in idx and "gy_raw" in idx and "gz_raw" in idx
    roll = v("roll_deg")
    if math.isnan(roll):
        roll = v("roll_dmp_deg")
    pitch = v("pitch_deg")
    if math.isnan(pitch):
        pitch = v("pitch_dmp_deg")
    yaw = v("yaw_deg")
    if math.isnan(yaw):
        yaw = v("yaw_dmp_deg")

    t_ms = v("time_ms")
    dmp_ok_raw = v("dmp_ok")
    if math.isnan(dmp_ok_raw):
        dmp_ok = 1 if "dmp_ok" not in idx else 0
    else:
        dmp_ok = 1 if int(dmp_ok_raw) != 0 else 0

    return {
        "time_ms": int(t_ms) if not math.isnan(t_ms) else 0,
        "lat": v("lat"),
        "lon": v("lon"),
        "fix": v("fix"),
        "sats": v("sats"),
        "dmp_ok": dmp_ok,
        "ax": v("ax_raw") if accel_is_raw else v("ax"),
        "ay": v("ay_raw") if accel_is_raw else v("ay"),
        "az": v("az_raw") if accel_is_raw else v("az"),
        "gx": v("gx_raw") if gyro_is_raw else v("gx"),
        "gy": v("gy_raw") if gyro_is_raw else v("gy"),
        "gz": v("gz_raw") if gyro_is_raw else v("gz"),
        "roll": roll,
        "pitch": pitch,
        "yaw": yaw,
        "accel_is_raw": accel_is_raw,
        "gyro_is_raw": gyro_is_raw,
    }


def ingest_sample(sample: dict) -> None:
    """Обновление треков и фильтра по одному измерению (serial или CSV)."""
    with _lock:
        state["latest"] = sample
        state["has_data"] = True

        lat = sample["lat"]
        lon = sample["lon"]
        roll = sample["roll"]
        pitch = sample["pitch"]
        yaw = sample["yaw"]
        dmp_ok = int(sample.get("dmp_ok", 1))
        fix = sample.get("fix", float("nan"))
        sats = sample.get("sats", float("nan"))

        if state["lat0"] is None and not (math.isnan(lat) or math.isnan(lon)):
            state["lat0"] = lat
            state["lon0"] = lon
            state["p_enu"] = np.zeros(3, dtype=float)
            state["v_enu"] = np.zeros(3, dtype=float)
            state["gnss_track"].append([lat, lon])
            state["ins_track"].append([lat, lon])
            state["kalman_track"].append([lat, lon])
            state["last_t_ms"] = sample["time_ms"]
            cfg = FilterConfig(
                dt=0.1,
                sigma_acc=0.3,
                sigma_gnss_pos=3.0,
                sigma_gnss_vel=1.0,
                use_gnss_velocity=False,
            )
            x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
            state["kf"] = INSGNSSKalmanFilter(cfg, x0)
            return

        if not (math.isnan(lat) or math.isnan(lon)):
            state["gnss_track"].append([lat, lon])
            if len(state["gnss_track"]) > 5000:
                state["gnss_track"] = state["gnss_track"][-5000:]

        if state["lat0"] is None:
            return

        t_ms = sample["time_ms"]
        if state["last_t_ms"] is None:
            state["last_t_ms"] = t_ms
        dt = float(np.clip((t_ms - state["last_t_ms"]) / 1000.0, 1e-3, 1.0))
        state["last_t_ms"] = t_ms

        ax_b, ay_b, az_b = float(sample["ax"]), float(sample["ay"]), float(sample["az"])
        gx_b, gy_b, gz_b = float(sample["gx"]), float(sample["gy"]), float(sample["gz"])
        if sample.get("accel_is_raw", False):
            # Масштабирование MPU6050 raw accel -> m/s^2 для FS_2G.
            ax_b *= ACC_SCALE_RAW
            ay_b *= ACC_SCALE_RAW
            az_b *= ACC_SCALE_RAW
        if sample.get("gyro_is_raw", False):
            # Масштабирование MPU6050 raw gyro -> rad/s для FS_250DPS.
            gx_b *= GYRO_SCALE_RAW
            gy_b *= GYRO_SCALE_RAW
            gz_b *= GYRO_SCALE_RAW
        # Для не-raw данных ожидаем: accel в м/с^2, gyro в рад/с или deg/s (для ZUPT ниже достаточно только порядка величин).

        acc_norm = float(np.linalg.norm([ax_b, ay_b, az_b])) if np.isfinite([ax_b, ay_b, az_b]).all() else float("nan")
        acc_minus_g = acc_norm - GRAVITY if math.isfinite(acc_norm) else float("nan")

        dbg = state["last_debug"]
        dbg["dt"] = dt
        dbg["acc_norm"] = acc_norm
        dbg["acc_minus_g"] = acc_minus_g
        dbg["dmp_ok"] = dmp_ok
        dbg["fix"] = float(fix)
        dbg["sats"] = float(sats)

        # Если DMP пакет невалидный или есть NaN в ключевых полях, ИНС-интеграцию пропускаем.
        imu_fields_ok = not any(math.isnan(x) for x in (ax_b, ay_b, az_b, roll, pitch, yaw))
        if dmp_ok == 0 or not imu_fields_ok:
            state["skipped_imu"] += 1
            dbg["a_n"] = float("nan")
            dbg["a_e"] = float("nan")
            dbg["v_n_ins"] = float(state["v_enu"][1])
            dbg["v_e_ins"] = float(state["v_enu"][0])
            # Kalman по IMU не прогнозируем, но позиция GNSS может корректировать фильтр.
            kf = state["kf"]
            if kf is not None and not (math.isnan(lat) or math.isnan(lon)):
                p_gnss = ll_to_enu(lat, lon, state["lat0"], state["lon0"])
                kf.update(p_gnss)
                p_kf = kf.get_position()
                lat_kf, lon_kf = enu_to_ll(float(p_kf[0]), float(p_kf[1]), state["lat0"], state["lon0"])
                state["kalman_track"].append([lat_kf, lon_kf])
                if len(state["kalman_track"]) > 5000:
                    state["kalman_track"] = state["kalman_track"][-5000:]
            return

        # Yaw only from Arduino/DMP. No GNSS course correction.
        c_nb = euler_to_c_nb(roll, pitch, yaw)
        a_neu = c_nb @ np.array([ax_b, ay_b, az_b], dtype=float)
        a_neu = a_neu + np.array([0.0, 0.0, -GRAVITY], dtype=float)
        a_n = float(a_neu[0])
        a_e = float(a_neu[1])
        a_u = float(a_neu[2])

        # Deadband for low horizontal acceleration.
        if abs(a_n) < 0.03:
            a_n = 0.0
        if abs(a_e) < 0.03:
            a_e = 0.0

        a_h = float(np.hypot(a_n, a_e))
        if a_h > 15.0:
            state["skipped_imu"] += 1
            dbg["a_n"] = a_n
            dbg["a_e"] = a_e
            dbg["v_n_ins"] = float(state["v_enu"][1])
            dbg["v_e_ins"] = float(state["v_enu"][0])
            return

        # ENU integration in meters (do not mix degrees/meters in state).
        a_enu = np.array([a_e, a_n, a_u], dtype=float)
        state["v_enu"] = state["v_enu"] + a_enu * dt
        state["p_enu"] = state["p_enu"] + state["v_enu"] * dt

        # Simple ZUPT: near-stationary by GNSS + IMU indicators.
        gnss_stationary = False
        if len(state["gnss_track"]) >= 2:
            lat_prev, lon_prev = state["gnss_track"][-2]
            d_enu = ll_to_enu(lat, lon, lat_prev, lon_prev) if not (math.isnan(lat) or math.isnan(lon)) else np.zeros(3)
            gnss_stationary = float(np.hypot(d_enu[0], d_enu[1])) < 0.3
        gyro_norm = float(np.linalg.norm([gx_b, gy_b, gz_b])) if np.isfinite([gx_b, gy_b, gz_b]).all() else float("inf")
        # Для физических (не raw) CSV гироскоп часто в deg/s; для ZUPT приводим к rad/s эвристически.
        if (not sample.get("gyro_is_raw", False)) and gyro_norm > 2.0:
            gyro_norm = float(np.deg2rad(gyro_norm))
        if gnss_stationary and abs(acc_minus_g) < 0.25 and gyro_norm < 0.15:
            state["v_enu"][0] *= 0.9
            state["v_enu"][1] *= 0.9

        lat_i, lon_i = enu_to_ll(float(state["p_enu"][0]), float(state["p_enu"][1]), state["lat0"], state["lon0"])
        state["ins_track"].append([lat_i, lon_i])
        if len(state["ins_track"]) > 5000:
            state["ins_track"] = state["ins_track"][-5000:]

        kf = state["kf"]
        if kf is not None:
            kf.predict(a_enu, dt=dt)
            if not (math.isnan(lat) or math.isnan(lon)):
                p_gnss = ll_to_enu(lat, lon, state["lat0"], state["lon0"])
                kf.update(p_gnss)
            p_kf = kf.get_position()
            lat_kf, lon_kf = enu_to_ll(float(p_kf[0]), float(p_kf[1]), state["lat0"], state["lon0"])
            state["kalman_track"].append([lat_kf, lon_kf])
            if len(state["kalman_track"]) > 5000:
                state["kalman_track"] = state["kalman_track"][-5000:]

        dbg["a_n"] = a_n
        dbg["a_e"] = a_e
        dbg["v_n_ins"] = float(state["v_enu"][1])
        dbg["v_e_ins"] = float(state["v_enu"][0])


def process_text_line(text: str) -> None:
    if not text or text.startswith("#"):
        return
    text_l = text.strip().lower().lstrip("\ufeff")
    if "time_ms" in text_l and "," in text_l and ("lat" in text_l and "lon" in text_l):
        header = [h.strip().lower() for h in text_l.split(",")]
        with _lock:
            state["header_idx"] = {h: i for i, h in enumerate(header)}
            state["arduino_header"] = [h.strip() for h in text.split(",")]
        return

    with _lock:
        idx = state["header_idx"]
    parts = [p.strip() for p in text.split(",")]
    if idx is None:
        # Fallback для потока, где заголовок потерян/поврежден.
        if len(parts) >= 21:
            fallback_header = [
                "time_ms", "lat", "lon", "alt_m", "speed_mps", "sats", "fix",
                "ax", "ay", "az", "gx_raw", "gy_raw", "gz_raw",
                "roll_dmp_deg", "pitch_dmp_deg", "yaw_dmp_deg",
                "qw", "qx", "qy", "qz", "dmp_ok",
            ]
            idx = {h: i for i, h in enumerate(fallback_header)}
        else:
            return

    sample = build_sample_from_parts(parts, idx)
    if sample is not None:
        with _lock:
            state["arduino_rows"].append(parts)
            if len(state["arduino_rows"]) > 20000:
                state["arduino_rows"] = state["arduino_rows"][-20000:]
        ingest_sample(sample)
    else:
        with _lock:
            state["parse_errors"] += 1


def read_serial_thread(ser: serial.Serial):
    while getattr(ser, "_running", True) and ser.is_open:
        try:
            line = ser.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").strip()
            process_text_line(text)
        except Exception:
            pass


def replay_csv_thread(csv_path: str, replay_speed: float) -> None:
    """Те же правила парсинга, что и для COM; пауза по дельте time_ms (ускорение replay_speed)."""
    try:
        with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
            for raw in f:
                text = raw.strip()
                if not text:
                    continue
                with _lock:
                    t_prev = state["latest"]["time_ms"] if state["latest"] else None
                process_text_line(text)
                with _lock:
                    t_curr = state["latest"]["time_ms"] if state["latest"] else None
                if replay_speed > 0 and t_prev is not None and t_curr is not None and t_curr > t_prev:
                    delay = (t_curr - t_prev) / 1000.0 / replay_speed
                    if delay > 0:
                        time.sleep(min(delay, 2.0))
    finally:
        with _lock:
            state["replay_finished"] = True
        print("Воспроизведение CSV завершено:", csv_path)


@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Realtime OSM: GNSS vs INS vs Kalman</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    .panel {
      position: absolute; top: 10px; left: 10px; z-index: 1000;
      background: rgba(255,255,255,0.9); padding: 8px 10px; border-radius: 8px; font-family: sans-serif;
      box-shadow: 0 1px 5px rgba(0,0,0,0.2);
    }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <div><b>Realtime OSM</b></div>
    <div style="color:#2e7d32;">GNSS</div>
    <div style="color:#c62828;">INS</div>
    <div style="color:#1565c0;">Kalman (INS+GNSS)</div>
    <div id="status">ожидание данных...</div>
    <div id="debug" style="font-size:12px; margin-top:6px; white-space:pre-line;"></div>
  </div>
<script>
const map = L.map('map').setView([55.75, 37.61], 14);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  maxZoom: 19,
  attribution: '&copy; OpenStreetMap contributors'
}).addTo(map);

const gnssLine = L.polyline([], {color: '#2e7d32', weight: 3}).addTo(map);
const insLine = L.polyline([], {color: '#c62828', weight: 3}).addTo(map);
const kalmanLine = L.polyline([], {color: '#1565c0', weight: 3}).addTo(map);
const gnssMarker = L.circleMarker([0,0], {radius: 5, color: '#2e7d32'}).addTo(map);
const insMarker = L.circleMarker([0,0], {radius: 5, color: '#c62828'}).addTo(map);
const kalmanMarker = L.circleMarker([0,0], {radius: 5, color: '#1565c0'}).addTo(map);
let centered = false;

async function tick() {
  try {
    const r = await fetch('/data?t=' + Date.now());
    const d = await r.json();
    document.getElementById('status').innerText = d.status;
    document.getElementById('debug').innerText =
      `dt=${(d.dt ?? 0).toFixed(3)}s  dmp_ok=${d.dmp_ok}  fix=${d.fix} sats=${d.sats}\n` +
      `|a|=${Number.isFinite(d.acc_norm) ? d.acc_norm.toFixed(3) : 'nan'}  |a|-g=${Number.isFinite(d.acc_minus_g) ? d.acc_minus_g.toFixed(3) : 'nan'}\n` +
      `aN=${Number.isFinite(d.a_n) ? d.a_n.toFixed(3) : 'nan'}  aE=${Number.isFinite(d.a_e) ? d.a_e.toFixed(3) : 'nan'}\n` +
      `vN=${(d.v_n_ins ?? 0).toFixed(3)}  vE=${(d.v_e_ins ?? 0).toFixed(3)}  skipped=${d.skipped_imu}  parseErr=${d.parse_errors}`;

    gnssLine.setLatLngs(d.gnss_track);
    insLine.setLatLngs(d.ins_track);
    kalmanLine.setLatLngs(d.kalman_track);

    if (d.gnss_track.length > 0) {
      const p = d.gnss_track[d.gnss_track.length - 1];
      gnssMarker.setLatLng(p);
      if (!centered) { map.setView(p, 17); centered = true; }
    }
    if (d.ins_track.length > 0) {
      const p2 = d.ins_track[d.ins_track.length - 1];
      insMarker.setLatLng(p2);
    }
    if (d.kalman_track.length > 0) {
      const p3 = d.kalman_track[d.kalman_track.length - 1];
      kalmanMarker.setLatLng(p3);
    }
  } catch (e) {}
}
setInterval(tick, 500);
tick();
</script>
</body>
</html>
"""


@app.get("/data")
def data():
    with _lock:
        gnss = list(state["gnss_track"])
        ins = list(state["ins_track"])
        kalman = list(state["kalman_track"])
        mode = state["mode"]
        replay_finished = state["replay_finished"]
        rpath = state["replay_path"]
        has_data = state["has_data"]
        dbg = dict(state["last_debug"])
        skipped_imu = int(state["skipped_imu"])
        parse_errors = int(state["parse_errors"])

    if mode == "replay":
        base = "воспроизведение CSV"
        if rpath:
            base += ": " + os.path.basename(rpath)
        if replay_finished:
            status = base + " (завершено)"
        else:
            status = base + " (идёт)"
    elif has_data:
        status = "идёт приём с COM"
    else:
        status = "ожидание данных с COM"

    fix_val = dbg.get("fix", float("nan"))
    sats_val = dbg.get("sats", float("nan"))
    fix_json = float(fix_val) if isinstance(fix_val, (int, float)) and math.isfinite(float(fix_val)) else None
    sats_json = float(sats_val) if isinstance(sats_val, (int, float)) and math.isfinite(float(sats_val)) else None

    return jsonify(
        {
            "gnss_track": gnss,
            "ins_track": ins,
            "kalman_track": kalman,
            "status": status,
            "mode": mode,
            "dt": float(dbg.get("dt", 0.0)),
            "acc_norm": float(dbg.get("acc_norm", float("nan"))),
            "acc_minus_g": float(dbg.get("acc_minus_g", float("nan"))),
            "a_n": float(dbg.get("a_n", float("nan"))),
            "a_e": float(dbg.get("a_e", float("nan"))),
            "v_n_ins": float(dbg.get("v_n_ins", 0.0)),
            "v_e_ins": float(dbg.get("v_e_ins", 0.0)),
            "dmp_ok": int(dbg.get("dmp_ok", 1)),
            "fix": fix_json,
            "sats": sats_json,
            "skipped_imu": skipped_imu,
            "parse_errors": parse_errors,
        }
    )


def save_snapshot_html():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return save_snapshot_html_with_ts(ts)


def save_snapshot_csv(ts: str):
    with _lock:
        header = list(state["arduino_header"]) if state["arduino_header"] else None
        rows = [list(r) for r in state["arduino_rows"]]

    if not rows:
        print("Нет исходных данных Arduino, CSV сохранять нечего.")
        return None

    out_name = os.path.join(SCRIPT_DIR, f"realtime_osm_snapshot_{ts}.csv")
    with open(out_name, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)
    print("CSV сохранен:", out_name)
    return out_name


def save_snapshot_bundle():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_path = save_snapshot_html_with_ts(ts)
    csv_path = save_snapshot_csv(ts)
    if html_path and csv_path:
        print("Снимок сохранен: HTML + CSV")
    elif html_path:
        print("Снимок сохранен: только HTML")
    elif csv_path:
        print("Снимок сохранен: только CSV")


def save_snapshot_html_with_ts(ts: str):
    with _lock:
        gnss = list(state["gnss_track"])
        ins = list(state["ins_track"])
        kalman = list(state["kalman_track"])
    if not gnss and not ins and not kalman:
        print("Треки пустые, карту сохранять нечего.")
        return None

    center = gnss[-1] if gnss else (kalman[-1] if kalman else ins[-1])
    out_name = os.path.join(SCRIPT_DIR, f"realtime_osm_snapshot_{ts}.html")
    gnss_js = str(gnss)
    ins_js = str(ins)
    kalman_js = str(kalman)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Realtime OSM Snapshot</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>html, body, #map {{ height: 100%; margin: 0; }}</style>
</head>
<body>
<div id="map"></div>
<script>
  const map = L.map('map').setView([{center[0]}, {center[1]}], 17);
  L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }}).addTo(map);
  const gnss = {gnss_js};
  const ins = {ins_js};
  const kalman = {kalman_js};
  if (gnss.length) {{
    L.polyline(gnss, {{color:'#2e7d32', weight:3}}).addTo(map);
    L.circleMarker(gnss[gnss.length-1], {{radius:5, color:'#2e7d32'}}).addTo(map);
  }}
  if (ins.length) {{
    L.polyline(ins, {{color:'#c62828', weight:3}}).addTo(map);
    L.circleMarker(ins[ins.length-1], {{radius:5, color:'#c62828'}}).addTo(map);
  }}
  if (kalman.length) {{
    L.polyline(kalman, {{color:'#1565c0', weight:3}}).addTo(map);
    L.circleMarker(kalman[kalman.length-1], {{radius:5, color:'#1565c0'}}).addTo(map);
  }}
</script>
</body>
</html>"""
    with open(out_name, "w", encoding="utf-8") as f:
        f.write(html)
    print("Карта сохранена:", out_name)
    return out_name


def main():
    parser = argparse.ArgumentParser(description="Realtime OSM: GNSS / INS / Kalman с COM или CSV.")
    parser.add_argument(
        "--file",
        metavar="PATH",
        help="Воспроизвести CSV вместо COM (тот же парсер, карта http://127.0.0.1:5000)",
    )
    parser.add_argument(
        "--replay-speed",
        type=float,
        default=20.0,
        help="Ускорение воспроизведения по меткам time_ms (1.0 ≈ реальное время). По умолчанию 20.",
    )
    parser.add_argument("--port", type=str, default=None, help="COM-порт (если не задан — выбор из списка)")
    args = parser.parse_args()

    ser = None
    if args.file:
        csv_path = args.file if os.path.isabs(args.file) else os.path.join(SCRIPT_DIR, args.file)
        if not os.path.isfile(csv_path):
            print("Файл не найден:", csv_path)
            return
        with _lock:
            state["mode"] = "replay"
            state["replay_path"] = csv_path
            state["replay_finished"] = False
        threading.Thread(
            target=replay_csv_thread,
            args=(csv_path, args.replay_speed),
            daemon=True,
        ).start()
        print("Режим CSV:", csv_path)
    else:
        with _lock:
            state["mode"] = "serial"
        ports = list(serial.tools.list_ports.comports())
        guess = None
        for p in ports:
            desc = p.description or ""
            if "Arduino" in desc or "CH340" in desc or "USB" in desc:
                guess = p.device
                break
        if not guess and ports:
            guess = ports[0].device
        if not guess:
            print("COM-порт не найден. Укажите CSV: --file path.csv")
            return

        port = args.port or input(f"Порт (Enter = {guess}): ").strip() or guess
        try:
            ser = serial.Serial(port, BAUD, timeout=0.05)
        except Exception as e:
            print("Ошибка открытия порта:", e)
            return

        ser._running = True
        threading.Thread(target=read_serial_thread, args=(ser,), daemon=True).start()
        print("Чтение COM запущено:", port)

    print("Откройте: http://127.0.0.1:5000")
    try:
        webbrowser.open("http://127.0.0.1:5000", new=2)
    except Exception:
        pass

    try:
        app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
    finally:
        save_snapshot_bundle()
        if ser is not None:
            ser._running = False
            time.sleep(0.1)
            ser.close()


if __name__ == "__main__":
    main()

