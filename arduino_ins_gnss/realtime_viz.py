# -*- coding: utf-8 -*-
"""
Визуализация положения системы в реальном времени по данным с Arduino.
Читает COM-порт, парсит CSV (time_ms, lat, lon, alt, ..., roll_deg, pitch_deg, yaw_deg),
показывает 3D ориентацию (курсовертикаль) и координаты ГНСС.
Закрытие окна или Ctrl+C — выход.
"""
# Зависимости: pip install pyserial matplotlib numpy  (именно pyserial, не "serial")
import math
import threading
import serial
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Данные с порта (общие для потока и анимации)
_data_lock = threading.Lock()
latest = {
    "time_ms": 0,
    "lat": float("nan"),
    "lon": float("nan"),
    "alt_m": float("nan"),
    "speed_mps": float("nan"),
    "sats": 0,
    "fix": 0,
    "ax": float("nan"),
    "ay": float("nan"),
    "az": float("nan"),
    "roll_deg": 0.0,
    "pitch_deg": 0.0,
    "yaw_deg": 0.0,
    "ok": False,
}

BAUD = 115200
ARROW_LEN = 1.0
# Масштаб стрелки ускорения в навигационной СК (NED): длина = |a_nav| * ACC_SCALE (м/с² -> единицы оси)
ACC_ARROW_SCALE = 0.05


def parse_float(s):
    s = (s or "").strip().lower()
    if s in ("", "nan", "nan(ind)"):
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def euler_rotation_matrix(roll_deg, pitch_deg, yaw_deg):
    """Матрица поворота из углов Эйлера ZYX (град): связ. СК -> нав. СК (NED)."""
    r = math.radians(roll_deg)
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    R = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                  cp * cr],
    ])
    return R


def read_serial_thread(ser):
    """Поток: читает строки, парсит CSV, обновляет latest."""
    global latest
    while getattr(ser, "_running", True) and ser.is_open:
        try:
            line = ser.readline()
            if not line:
                continue
            text = line.decode("utf-8", errors="replace").strip()
            if not text or text.startswith("#"):
                continue
            if text.startswith("time_ms,"):
                continue
            parts = [p.strip() for p in text.split(",")]
            if len(parts) < 13:
                continue
            with _data_lock:
                try:
                    latest["time_ms"] = int(parts[0])
                except (ValueError, IndexError):
                    pass
                latest["lat"] = parse_float(parts[1])
                latest["lon"] = parse_float(parts[2])
                latest["alt_m"] = parse_float(parts[3])
                latest["speed_mps"] = parse_float(parts[4])
                latest["sats"] = int(parts[5]) if parts[5].strip().isdigit() else 0
                latest["fix"] = int(parts[6]) if parts[6].strip().isdigit() else 0
                latest["ax"] = parse_float(parts[7])
                latest["ay"] = parse_float(parts[8])
                latest["az"] = parse_float(parts[9])
                if len(parts) >= 16:
                    latest["roll_deg"] = parse_float(parts[13])
                    latest["pitch_deg"] = parse_float(parts[14])
                    latest["yaw_deg"] = parse_float(parts[15])
                latest["ok"] = True
        except Exception:
            pass


def main():
    ports = list(serial.tools.list_ports.comports())
    arduino_port = None
    for p in ports:
        if "Arduino" in (p.description or "") or "CH340" in (p.description or "") or "USB" in (p.description or ""):
            arduino_port = p.device
            break
    if not arduino_port and ports:
        arduino_port = ports[0].device

    if not arduino_port:
        print("COM-порт не найден. Подключите Arduino.")
        return

    port = input(f"Порт (Enter = {arduino_port}): ").strip() or arduino_port
    try:
        ser = serial.Serial(port, BAUD, timeout=0.05)
    except Exception as e:
        print("Ошибка открытия порта:", e)
        return

    ser._running = True
    thread = threading.Thread(target=read_serial_thread, args=(ser,), daemon=True)
    thread.start()

    fig = plt.figure(figsize=(10, 6))
    ax3d = fig.add_subplot(121, projection="3d")
    ax2d = fig.add_subplot(122)
    ax2d.set_axis_off()
    info_text = ax2d.text(0.05, 0.95, "", transform=ax2d.transAxes,
                          fontsize=11, verticalalignment="top", family="monospace",
                          bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    # 3D: три стрелки осей (X,Y,Z) — линии от 0 до R @ e_i
    lines = []
    for _ in range(3):
        line, = ax3d.plot([0, 1], [0, 0], [0, 0], linewidth=2.5)
        lines.append(line)
    colors = ["#c0392b", "#27ae60", "#2980b9"]  # R, G, B
    for i, line in enumerate(lines):
        line.set_color(colors[i])
    # Стрелка ускорения в навигационной СК (NED)
    acc_line, = ax3d.plot([0, 0], [0, 0], [0, 0], linewidth=3, color="#9b59b6", label="a (навиг.)")
    lines.append(acc_line)
    ax3d.set_xlim(-ARROW_LEN, ARROW_LEN)
    ax3d.set_ylim(-ARROW_LEN, ARROW_LEN)
    ax3d.set_zlim(-ARROW_LEN, ARROW_LEN)
    ax3d.set_xlabel("Север (N)")
    ax3d.set_ylabel("Восток (E)")
    ax3d.set_zlabel("Вниз (D)")
    ax3d.set_title("Нав. СК (NED): ориентация и ускорение")

    def update(_):
        with _data_lock:
            roll = latest["roll_deg"]
            pitch = latest["pitch_deg"]
            yaw = latest["yaw_deg"]
            ax_b = latest["ax"]
            ay_b = latest["ay"]
            az_b = latest["az"]
            lat = latest["lat"]
            lon = latest["lon"]
            alt = latest["alt_m"]
            speed = latest["speed_mps"]
            sats = latest["sats"]
            fix = latest["fix"]
            time_ms = latest["time_ms"]

        if math.isnan(roll):
            roll = 0.0
        if math.isnan(pitch):
            pitch = 0.0
        if math.isnan(yaw):
            yaw = 0.0

        R = euler_rotation_matrix(roll, pitch, yaw)
        origins = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        ends = (R * ARROW_LEN).T  # строки ends = столбцы R (направления осей X,Y,Z)
        for i in range(3):
            lines[i].set_data_3d(
                [origins[i, 0], ends[i, 0]],
                [origins[i, 1], ends[i, 1]],
                [origins[i, 2], ends[i, 2]],
            )

        # Ускорение в навигационной СК (NED): a_nav = R^T @ a_body
        if not (math.isnan(ax_b) or math.isnan(ay_b) or math.isnan(az_b)):
            a_body = np.array([ax_b, ay_b, az_b])
            a_nav = R.T @ a_body  # проекции на оси нав. СК: Север, Восток, Вниз (м/с²)
            end_acc = a_nav * ACC_ARROW_SCALE
            acc_line.set_data_3d([0, end_acc[0]], [0, end_acc[1]], [0, end_acc[2]])
            ax_nav, ay_nav, az_nav = a_nav[0], a_nav[1], a_nav[2]
            acc_str = (
                f"Ускорение (навиг. NED), м/с²:\n"
                f"  a_N (Север): {ax_nav:+.2f}\n"
                f"  a_E (Восток): {ay_nav:+.2f}\n"
                f"  a_D (Вниз): {az_nav:+.2f}\n"
            )
        else:
            acc_line.set_data_3d([0, 0], [0, 0], [0, 0])
            acc_str = "Ускорение (навиг.): —\n"

        lat_s = f"{lat:.6f}" if not math.isnan(lat) else "—"
        lon_s = f"{lon:.6f}" if not math.isnan(lon) else "—"
        alt_s = f"{alt:.1f} м" if not math.isnan(alt) else "—"
        spd_s = f"{speed:.2f} м/с" if not math.isnan(speed) else "—"
        info_text.set_text(
            f"Время: {time_ms} мс\n\n"
            f"Крен:   {roll:+.1f}°\n"
            f"Тангаж: {pitch:+.1f}°\n"
            f"Рысканье: {yaw:+.1f}°\n\n"
            f"{acc_str}\n"
            f"Широта:  {lat_s}\n"
            f"Долгота: {lon_s}\n"
            f"Высота:  {alt_s}\n"
            f"Скорость: {spd_s}\n"
            f"Спутники: {sats}  Fix: {fix}"
        )
        return lines + [info_text]

    ani = FuncAnimation(fig, update, interval=50, blit=False)
    plt.tight_layout()
    print("Визуализация запущена. Закройте окно для выхода.")
    plt.show()
    ser._running = False
    ser.close()


if __name__ == "__main__":
    main()
