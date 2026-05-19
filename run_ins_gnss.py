# -*- coding: utf-8 -*-
"""Запуск обработки заезда: ищет ins_gnss_*.csv в папке, строит карту и графики."""
import os
import sys
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

def main():
    pattern = os.path.join(SCRIPT_DIR, "ins_gnss_*.csv")
    files = glob.glob(pattern)
    if not files:
        print("No ins_gnss_*.csv in:", SCRIPT_DIR)
        sys.exit(1)
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    csv_path = os.path.abspath(files[0])
    print("CSV:", csv_path)

    import check_ins_gnss_csv as check
    sys.argv = ["check_ins_gnss_csv.py", csv_path]
    check.main()

    import analyze_ins_csv as anal
    sys.argv = ["analyze_ins_csv.py", csv_path]
    anal.main()

    print("Done. Open ins_gnss_check_map.html and ins_gnss_analysis_map.html")

if __name__ == "__main__":
    main()
