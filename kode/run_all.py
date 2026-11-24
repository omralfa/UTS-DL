"""
Master reproduction script for UTS Deep Learning - Kelompok 4
Anggota:
  - Ruben Alfa Richie (101032300008, GitHub: rubenalfarichie) -> Task 1 & Task 2
  - Omar Al Farizy Hutagalung (101032300034, GitHub: omaralfa) -> Task 3 & Laporan

Menjalankan seluruh pipeline (EDA, Tugas 1, Tugas 2, Tugas 3, Verifikasi, dan Laporan PDF)
dengan satu perintah:
    python kode/run_all.py
"""
import argparse
import subprocess
import sys
import os

KODE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_step(script_name, extra_args=None):
    cmd = [sys.executable, os.path.join(KODE_DIR, script_name)]
    if extra_args:
        cmd.extend(extra_args)
    print("\n" + "=" * 72)
    print("RUNNING:", " ".join(cmd))
    print("=" * 72)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Reproduce all UTS Deep Learning results in one command.")
    parser.add_argument("--skip-report", action="store_true", help="Skip PDF report regeneration")
    args = parser.parse_args()

    run_step("eda_figures.py")
    run_step("task1_classification.py")
    run_step("task2_rul_regression.py")
    run_step("task3_clustering.py")
    run_step("verify_submission.py")
    if not args.skip_report:
        run_step("generate_report.py")
    print("\nALL STEPS COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
