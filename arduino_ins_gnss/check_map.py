# -*- coding: utf-8 -*-
"""
Проверка потока Arduino -> Serial -> карта OSM.

Показывает отдельно:
- GNSS (спутник)
- INS (интеграция IMU в ENU через матрицу body->nav)

Калмана здесь нет.

Запуск:
  python check_map.py
  python check_map.py --port COM5
"""
from __future__ import annotations

import argparse
import math
import threading
import time
import webbrowser

import serial
import serial.tools.list_ports
from flask import Flask, jsonify


app = Flask(__name__)
_lock = threading.Lock()

state = {
    "header_idx": None,
    "rows_total": 0,
    "rows_ok": 0,
    "rows_bad": 0,
    "last_line": "",
    "last_error": "",
    "gnss_track": [],
    "ins_track": [],
    "last_gnss_point": None,
    "last_ins_point": None,
    "lat0": None,
    "lon0": None,
    "p_enu": [0.0, 0.0, 0.0],  # [E, N, U], meters
    "v_enu": [0.0, 0.0, 0.0],  # [E, N, U], m/s
    "last_t_ms": None,
    "dt_last": 0.0,
    "a_n_last": 0.0,
    "a_e_last": 0.0,
}


def parse_float(value: str) -> float:
    s = (value or "").strip().lower()
    if s in ("", "nan", "nan(ind)"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def detect_port(explicit_port: str | None) -> str | None:
    if explicit_port:
        return explicit_port
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    for p in ports:
        desc = (p.description or "").lower()
        if "arduino" in desc or "ch340" in desc or "usb" in desc:
            return p.device
    return ports[0].device


def ll_to_enu(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    r = 6378137.0
    d_lat = math.radians(lat - lat0)
    d_lon = math.radians(lon - lon0)
    north = d_lat * r
    east = d_lon * r * math.cos(math.radians(lat0))
    return east, north


def enu_to_ll(east: float, north: float, lat0: float, lon0: float) -> tuple[float, float]:
    r = 6378137.0
    lat = lat0 + math.degrees(north / r)
    lon = lon0 + math.degrees(east / (r * math.cos(math.radians(lat0))))
    return lat, lon


def euler_to_c_nb(roll_deg: float, pitch_deg: float, yaw_deg: float) -> list[list[float]]:
    # body -> nav(NEU)
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    sr, cr = math.sin(r), math.cos(r)
    sp, cp = math.sin(p), math.cos(p)
    sy, cy = math.sin(y), math.cos(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def mat_vec_mul_3x3(m: list[list[float]], v: list[float]) -> list[float]:
    return [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]


def update_header_if_present(line: str) -> bool:
    line_l = line.lower().lstrip("\ufeff")
    if "time_ms" not in line_l or "lat" not in line_l or "lon" not in line_l:
        return False
    cols = [x.strip().lower() for x in line_l.split(",")]
    with _lock:
        state["header_idx"] = {name: i for i, name in enumerate(cols)}
    return True


def extract_lat_lon(parts: list[str], idx: dict[str, int] | None) -> tuple[float, float]:
    if idx is not None and "lat" in idx and "lon" in idx:
        i_lat = idx["lat"]
        i_lon = idx["lon"]
        if i_lat < len(parts) and i_lon < len(parts):
            return parse_float(parts[i_lat]), parse_float(parts[i_lon])
    if len(parts) >= 3:
        # fallback: time_ms,lat,lon,...
        return parse_float(parts[1]), parse_float(parts[2])
    return float("nan"), float("nan")


def extract_sample(parts: list[str], idx: dict[str, int] | None) -> dict:
    def get_value(name: str, fallback_pos: int | None = None) -> float:
        if idx is not None and name in idx and idx[name] < len(parts):
            return parse_float(parts[idx[name]])
        if fallback_pos is not None and fallback_pos < len(parts):
            return parse_float(parts[fallback_pos])
        return float("nan")

    # Expected stream:
    # time_ms,lat,lon,alt_m,speed_mps,sats,fix,ax,ay,az,gx_raw,gy_raw,gz_raw,roll_dmp_deg,pitch_dmp_deg,yaw_dmp_deg,...
    return {
        "time_ms": get_value("time_ms", 0),
        "lat": get_value("lat", 1),
        "lon": get_value("lon", 2),
        "ax": get_value("ax", 7),
        "ay": get_value("ay", 8),
        "az": get_value("az", 9),
        "roll": get_value("roll_dmp_deg", 13),
        "pitch": get_value("pitch_dmp_deg", 14),
        "yaw": get_value("yaw_dmp_deg", 15),
    }


def update_ins(sample: dict) -> None:
    gravity = 9.80665
    t_ms = sample["time_ms"]
    lat = sample["lat"]
    lon = sample["lon"]
    ax = sample["ax"]
    ay = sample["ay"]
    az = sample["az"]
    roll = sample["roll"]
    pitch = sample["pitch"]
    yaw = sample["yaw"]

    if not all(math.isfinite(v) for v in (t_ms, ax, ay, az, roll, pitch, yaw)):
        return

    if state["last_t_ms"] is None:
        state["last_t_ms"] = t_ms
        return

    dt = max(1e-3, min(1.0, (t_ms - state["last_t_ms"]) / 1000.0))
    state["last_t_ms"] = t_ms
    state["dt_last"] = dt

    c_nb = euler_to_c_nb(roll, pitch, yaw)
    a_neu = mat_vec_mul_3x3(c_nb, [ax, ay, az])
    # az включает g; в NEU U направлена вверх.
    a_neu[2] -= gravity
    a_n = a_neu[0]
    a_e = a_neu[1]
    a_u = a_neu[2]
    state["a_n_last"] = a_n
    state["a_e_last"] = a_e

    # Integrate in ENU: [E, N, U]
    state["v_enu"][0] += a_e * dt
    state["v_enu"][1] += a_n * dt
    state["v_enu"][2] += a_u * dt

    state["p_enu"][0] += state["v_enu"][0] * dt
    state["p_enu"][1] += state["v_enu"][1] * dt
    state["p_enu"][2] += state["v_enu"][2] * dt

    lat0 = state["lat0"]
    lon0 = state["lon0"]
    if lat0 is None or lon0 is None:
        return
    lat_i, lon_i = enu_to_ll(state["p_enu"][0], state["p_enu"][1], lat0, lon0)
    state["last_ins_point"] = [lat_i, lon_i]
    state["ins_track"].append([lat_i, lon_i])
    if len(state["ins_track"]) > 5000:
        state["ins_track"] = state["ins_track"][-5000:]


def process_line(raw_line: str) -> None:
    line = raw_line.strip()
    if not line:
        return
    if line.startswith("#"):
        return

    with _lock:
        state["rows_total"] += 1
        state["last_line"] = line[:300]

    if update_header_if_present(line):
        return

    parts = [x.strip() for x in line.split(",")]
    with _lock:
        idx = state["header_idx"]

    sample = extract_sample(parts, idx)
    lat = sample["lat"]
    lon = sample["lon"]

    if not (math.isfinite(lat) and math.isfinite(lon)):
        with _lock:
            state["rows_bad"] += 1
            state["last_error"] = "lat/lon is not finite"
        return
    if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
        with _lock:
            state["rows_bad"] += 1
            state["last_error"] = "lat/lon out of range"
        return

    with _lock:
        state["rows_ok"] += 1
        state["last_error"] = ""
        state["last_gnss_point"] = [lat, lon]
        state["gnss_track"].append([lat, lon])
        if len(state["gnss_track"]) > 5000:
            state["gnss_track"] = state["gnss_track"][-5000:]

        if state["lat0"] is None:
            state["lat0"] = lat
            state["lon0"] = lon
            state["p_enu"] = [0.0, 0.0, 0.0]
            state["v_enu"] = [0.0, 0.0, 0.0]
            state["last_t_ms"] = sample["time_ms"] if math.isfinite(sample["time_ms"]) else None

        # Если GNSS внезапно "прыгает" после старта, INS продолжаем от стартовой lat0/lon0.
        # Небольшая подстройка INS старта: если до этого не было INS-точек, привяжем к GNSS нулю.
        if not state["ins_track"] and state["lat0"] is not None and state["lon0"] is not None:
            state["ins_track"].append([state["lat0"], state["lon0"]])
            state["last_ins_point"] = [state["lat0"], state["lon0"]]

        update_ins(sample)


def serial_reader_loop(ser: serial.Serial) -> None:
    while getattr(ser, "_running", True) and ser.is_open:
        try:
            raw = ser.readline()
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            process_line(line)
        except Exception as e:
            with _lock:
                state["rows_bad"] += 1
                state["last_error"] = f"serial read error: {e}"
            time.sleep(0.02)


@app.get("/")
def index():
    return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>check_map</title>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    html, body, #map { height: 100%; margin: 0; }
    .panel {
      position: absolute; top: 10px; left: 10px; z-index: 1000;
      background: rgba(255,255,255,0.92); padding: 8px 10px; border-radius: 8px; font-family: sans-serif;
      box-shadow: 0 1px 5px rgba(0,0,0,0.2); max-width: 700px;
    }
    #diag { font-size: 12px; white-space: pre-line; margin-top: 6px; }
  </style>
</head>
<body>
  <div id="map"></div>
  <div class="panel">
    <div><b>check_map (GNSS vs INS)</b></div>
    <div id="status">waiting...</div>
    <div id="diag"></div>
  </div>
<script>
const map = L.map("map").setView([55.75, 37.61], 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19,
  attribution: "&copy; OpenStreetMap contributors"
}).addTo(map);

const gnssLine = L.polyline([], { color: "#2e7d32", weight: 3 }).addTo(map);
const insLine = L.polyline([], { color: "#c62828", weight: 3 }).addTo(map);
const gnssMarker = L.circleMarker([55.75, 37.61], { radius: 5, color: "#2e7d32" }).addTo(map);
const insMarker = L.circleMarker([55.75, 37.61], { radius: 5, color: "#c62828" }).addTo(map);
let centered = false;

async function tick() {
  try {
    const r = await fetch("/data?t=" + Date.now());
    const d = await r.json();
    document.getElementById("status").innerText = d.status;
    document.getElementById("diag").innerText =
      `rows_total=${d.rows_total}  rows_ok=${d.rows_ok}  rows_bad=${d.rows_bad}\\n` +
      `dt=${(d.dt_last || 0).toFixed(3)}  aN=${(d.a_n_last || 0).toFixed(3)}  aE=${(d.a_e_last || 0).toFixed(3)}\\n` +
      `last_error=${d.last_error || "-"}\\n` +
      `last_line=${d.last_line || "-"}`;
    gnssLine.setLatLngs(d.gnss_track);
    insLine.setLatLngs(d.ins_track);
    if (d.last_gnss_point) {
      gnssMarker.setLatLng(d.last_gnss_point);
      if (!centered) {
        map.setView(d.last_gnss_point, 17);
        centered = true;
      }
    }
    if (d.last_ins_point) {
      insMarker.setLatLng(d.last_ins_point);
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
        rows_total = int(state["rows_total"])
        rows_ok = int(state["rows_ok"])
        rows_bad = int(state["rows_bad"])
        last_line = str(state["last_line"])
        last_error = str(state["last_error"])
        gnss_track = list(state["gnss_track"])
        ins_track = list(state["ins_track"])
        last_gnss_point = state["last_gnss_point"]
        last_ins_point = state["last_ins_point"]
        dt_last = float(state["dt_last"])
        a_n_last = float(state["a_n_last"])
        a_e_last = float(state["a_e_last"])
    status = "идет прием данных" if rows_total > 0 else "ожидание данных serial"
    return jsonify(
        {
            "status": status,
            "rows_total": rows_total,
            "rows_ok": rows_ok,
            "rows_bad": rows_bad,
            "last_line": last_line,
            "last_error": last_error,
            "gnss_track": gnss_track,
            "ins_track": ins_track,
            "last_gnss_point": last_gnss_point,
            "last_ins_point": last_ins_point,
            "dt_last": dt_last,
            "a_n_last": a_n_last,
            "a_e_last": a_e_last,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple serial map checker for Arduino CSV stream.")
    parser.add_argument("--port", type=str, default=None, help="COM port, e.g. COM5")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default 115200)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Flask host")
    parser.add_argument("--web-port", type=int, default=5001, help="Flask port")
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()

    port = detect_port(args.port)
    if not port:
        print("COM port not found.")
        return

    try:
        ser = serial.Serial(port, args.baud, timeout=0.1)
    except Exception as e:
        print(f"Port open error ({port}): {e}")
        return

    ser._running = True
    threading.Thread(target=serial_reader_loop, args=(ser,), daemon=True).start()
    print(f"Serial reader started: {port} @ {args.baud}")

    url = f"http://{args.host}:{args.web_port}"
    print("Open:", url)
    if not args.no_browser:
        try:
            webbrowser.open(url, new=2)
        except Exception:
            pass

    try:
        app.run(host=args.host, port=args.web_port, debug=False, use_reloader=False)
    finally:
        ser._running = False
        time.sleep(0.1)
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()

