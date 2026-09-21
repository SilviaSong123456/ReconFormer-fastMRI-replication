import ast
import csv
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]

SOURCE_DIR = (
    PROJECT_DIR
    / "datasets"
    / "fastMRI"
    / "PD"
    / "val"
)

SINGLE_ROOT = (
    PROJECT_DIR
    / "datasets"
    / "single_volume"
)

SINGLE_VAL_DIR = (
    SINGLE_ROOT
    / "PD"
    / "val"
)

CHECKPOINT = (
    PROJECT_DIR
    / "checkpoints"
    / "F_X8_checkpoint.pth"
)

RESULT_DIR = (
    PROJECT_DIR
    / "results"
    / "final_199_x8"
)

LOG_DIR = RESULT_DIR / "per_volume_logs"
CSV_PATH = RESULT_DIR / "per_volume_metrics.csv"
SUMMARY_PATH = RESULT_DIR / "summary.txt"


SINGLE_VAL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# volume_files = sorted(SOURCE_DIR.glob("*.h5"))
volume_files = sorted(SOURCE_DIR.glob("*.h5"))

if not volume_files:
    raise FileNotFoundError(
        f"No H5 volumes were found in {SOURCE_DIR}"
    )

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

print(f"Found {len(volume_files)} volumes")
print(f"Python: {sys.executable}")
print(f"Checkpoint: {CHECKPOINT}")
print()


# Resume support: read volumes that have already succeeded.
completed = set()

if CSV_PATH.exists():
    with CSV_PATH.open("r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row.get("status") == "success":
                completed.add(row["volume"])

    print(f"Already completed: {len(completed)} volumes")


file_exists = CSV_PATH.exists()

csv_file = CSV_PATH.open("a", newline="", buffering=1)

fieldnames = [
    "volume",
    "number_of_slices",
    "val_loss",
    "nmse",
    "ssim",
    "psnr",
    "runtime_seconds",
    "status",
]

writer = csv.DictWriter(
    csv_file,
    fieldnames=fieldnames
)

if not file_exists:
    writer.writeheader()


def clear_temporary_directory():
    for path in SINGLE_VAL_DIR.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()


def extract_metric(output_text, metric_name):
    pattern = rf"'{metric_name}':\s*([-+0-9.eE]+)"
    match = re.search(pattern, output_text)

    if match is None:
        raise ValueError(
            f"Could not find {metric_name} in program output"
        )

    return float(match.group(1))


total = len(volume_files)
successful = 0
failed = 0

try:
    for index, volume_path in enumerate(
        volume_files,
        start=1
    ):
        volume_name = volume_path.name

        if volume_name in completed:
            print(
                f"[{index}/{total}] SKIP {volume_name} "
                f"(already completed)"
            )
            continue

        clear_temporary_directory()

        temporary_link = (
            SINGLE_VAL_DIR
            / volume_name
        )

        temporary_link.symlink_to(volume_path)

        log_path = (
            LOG_DIR
            / f"{volume_path.stem}.log"
        )

        command = [
            sys.executable,
            str(PROJECT_DIR / "main_recon_test.py"),
            "--phase",
            "test",
            "--model",
            "ReconFormer",
            "--challenge",
            "singlecoil",
            "--F_path",
            str(SINGLE_ROOT),
            "--test_dataset",
            "F",
            "--sequence",
            "PD",
            "--accelerations",
            "8",
            "--center-fractions",
            "0.04",
            "--checkpoint",
            str(CHECKPOINT),
            "--gpu",
            "0",
            "--verbose",
        ]

        print(
            f"[{index}/{total}] Running {volume_name}",
            flush=True
        )

        start_time = time.time()

        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = "0"

        process = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            capture_output=True,
            text=True,
        )

        runtime_seconds = time.time() - start_time

        combined_output = (
            process.stdout
            + "\n"
            + process.stderr
        )

        log_path.write_text(
            combined_output,
            encoding="utf-8"
        )

        if process.returncode == 0:
            try:
                val_loss = extract_metric(
                    combined_output,
                    "val_loss"
                )
                nmse = extract_metric(
                    combined_output,
                    "nmse"
                )
                ssim = extract_metric(
                    combined_output,
                    "ssim"
                )
                psnr = extract_metric(
                    combined_output,
                    "psnr"
                )

                slice_match = re.search(
                    r"(\d+)/\1",
                    combined_output
                )

                number_of_slices = (
                    int(slice_match.group(1))
                    if slice_match
                    else ""
                )

                writer.writerow({
                    "volume": volume_name,
                    "number_of_slices": number_of_slices,
                    "val_loss": val_loss,
                    "nmse": nmse,
                    "ssim": ssim,
                    "psnr": psnr,
                    "runtime_seconds": runtime_seconds,
                    "status": "success",
                })

                successful += 1

                print(
                    f"  SUCCESS | "
                    f"NMSE={nmse:.6f} | "
                    f"SSIM={ssim:.6f} | "
                    f"PSNR={psnr:.3f} | "
                    f"{runtime_seconds:.1f}s",
                    flush=True
                )

            except Exception as error:
                writer.writerow({
                    "volume": volume_name,
                    "number_of_slices": "",
                    "val_loss": "",
                    "nmse": "",
                    "ssim": "",
                    "psnr": "",
                    "runtime_seconds": runtime_seconds,
                    "status": f"parse_error: {error}",
                })

                failed += 1
                print(
                    f"  METRIC PARSE FAILED: {error}",
                    flush=True
                )

        else:
            writer.writerow({
                "volume": volume_name,
                "number_of_slices": "",
                "val_loss": "",
                "nmse": "",
                "ssim": "",
                "psnr": "",
                "runtime_seconds": runtime_seconds,
                "status": f"failed_exit_{process.returncode}",
            })

            failed += 1

            print(
                f"  FAILED | exit={process.returncode}",
                flush=True
            )
            print(
                f"  See log: {log_path}",
                flush=True
            )

finally:
    csv_file.close()
    clear_temporary_directory()


# Read all successful rows, including results from previous runs.
successful_rows = []

with CSV_PATH.open("r", newline="") as csv_file:
    reader = csv.DictReader(csv_file)

    for row in reader:
        if row["status"] == "success":
            successful_rows.append(row)


summary_lines = [
    "ReconFormer full validation evaluation",
    "=======================================",
    f"Source directory: {SOURCE_DIR}",
    f"Checkpoint: {CHECKPOINT}",
    f"Requested acceleration: 8x",
    f"Center fraction: 0.04",
    f"Total available volumes: {total}",
    f"Successful volumes: {len(successful_rows)}",
]

if successful_rows:
    for metric_name in [
        "val_loss",
        "nmse",
        "ssim",
        "psnr",
        "runtime_seconds",
    ]:
        values = np.array([
            float(row[metric_name])
            for row in successful_rows
        ])

        summary_lines.append(
            f"{metric_name}: "
            f"mean={values.mean():.8f}, "
            f"std={values.std(ddof=1):.8f}, "
            f"min={values.min():.8f}, "
            f"max={values.max():.8f}"
        )

summary_text = "\n".join(summary_lines)

SUMMARY_PATH.write_text(
    summary_text + "\n",
    encoding="utf-8"
)

print()
print(summary_text)
print()
print(f"CSV saved to: {CSV_PATH}")
print(f"Summary saved to: {SUMMARY_PATH}")