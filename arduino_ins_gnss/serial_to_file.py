import sys
from datetime import datetime

import serial
import serial.tools.list_ports


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
        sys.exit(1)

    out_name = f"ins_gnss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    port = input(f"Порт (Enter = {arduino_port}): ").strip() or arduino_port
    baud = 115200

    print(f"Открываю {port} @ {baud}, запись в {out_name}")
    print("Остановка: Ctrl+C")
    try:
        ser = serial.Serial(port, baud, timeout=0.5)
    except Exception as e:
        print("Ошибка открытия порта:", e)
        sys.exit(1)

    with open(out_name, "w", encoding="utf-8") as f:
        try:
            while True:
                line = ser.readline()
                if line:
                    try:
                        text = line.decode("utf-8", errors="replace").strip()
                    except Exception:
                        text = line.decode("cp1251", errors="replace").strip()
                    if not text:
                        continue
                    # Raw passthrough: пишем строку в файл без изменений.
                    line_out = text + "\n"
                    f.write(line_out)
                    f.flush()
                    print(line_out.strip()[:120] + ("..." if len(line_out) > 120 else ""))
        except KeyboardInterrupt:
            pass

    print(f"\nСохранено: {out_name}")

if __name__ == "__main__":
    main()
