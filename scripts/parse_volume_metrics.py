import argparse
import ast
import csv
import re
from pathlib import Path

import numpy as np


PATTERNS = {
    "zero": r"Zero-filled baseline:\s*(\{[^\n]+\})",
    "recon": r"ReconFormer:\s*(\{[^\n]+\})",
    "improve": (
        r"Improvement of ReconFormer over Zero-filled:"
        r"\s*(\{[^\n]+\})"
    ),
}


def parse_log(log_path):
    text = log_path.read_text(errors="replace")
    matches = {
        name: re.search(pattern, text)
        for name, pattern in PATTERNS.items()
    }

    if not all(matches.values()):
        return None

    zero = ast.literal_eval(matches["zero"].group(1))
    recon = ast.literal_eval(matches["recon"].group(1))
    improve = ast.literal_eval(matches["improve"].group(1))

    return {
        "volume": log_path.stem,
        "zero_nmse": zero["nmse"],
        "zero_ssim": zero["ssim"],
        "zero_psnr": zero["psnr"],
        "recon_val_loss": recon["val_loss"],
        "recon_nmse": recon["nmse"],
        "recon_ssim": recon["ssim"],
        "recon_psnr": recon["psnr"],
        "nmse_absolute_reduction": (
            improve["nmse_absolute_reduction"]
        ),
        "nmse_relative_reduction_percent": (
            improve["nmse_relative_reduction_percent"]
        ),
        "ssim_absolute_improvement": (
            improve["ssim_absolute_improvement"]
        ),
        "psnr_improvement_db": (
            improve["psnr_improvement_db"]
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Parse zero-filled and ReconFormer metrics "
            "from per-volume evaluation logs."
        )
    )
    parser.add_argument(
        "result_dir",
        type=Path,
        help=(
            "Evaluation directory containing "
            "per_volume_logs/"
        ),
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    log_dir = result_dir / "per_volume_logs"

    if not log_dir.exists():
        raise FileNotFoundError(
            f"Log directory not found: {log_dir}"
        )

    rows = []
    failed = []

    for log_path in sorted(log_dir.glob("*.log")):
        row = parse_log(log_path)

        if row is None:
            failed.append(log_path.stem)
        else:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No valid volume logs found in {log_dir}"
        )

    csv_path = (
        result_dir
        / "zero_filled_vs_reconformer.csv"
    )
    summary_path = (
        result_dir
        / "summary_corrected.txt"
    )

    fields = list(rows[0].keys())

    with csv_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "Corrected volume-level evaluation",
        "=" * 40,
        f"Successfully parsed volumes: {len(rows)}",
        f"Failed volumes: {len(failed)}",
        "",
    ]

    for metric in fields[1:]:
        values = np.asarray(
            [float(row[metric]) for row in rows]
        )
        lines.append(
            f"{metric}: "
            f"mean={values.mean():.8f}, "
            f"std={values.std(ddof=1):.8f}, "
            f"min={values.min():.8f}, "
            f"max={values.max():.8f}"
        )

    if failed:
        lines.extend([
            "",
            "Failed logs:",
            *failed,
        ])

    summary_path.write_text(
        "\n".join(lines) + "\n"
    )

    print(f"Parsed: {len(rows)}")
    print(f"Failed: {len(failed)}")
    print(f"CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
