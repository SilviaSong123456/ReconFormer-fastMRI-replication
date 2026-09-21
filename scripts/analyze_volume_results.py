import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon


def read_metrics(path):
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    return rows


def values(rows, column):
    return np.asarray(
        [float(row[column]) for row in rows],
        dtype=float,
    )


def check_volumes(x4_rows, x8_rows):
    x4_names = [row["volume"] for row in x4_rows]
    x8_names = [row["volume"] for row in x8_rows]

    if len(x4_names) != 199 or len(x8_names) != 199:
        raise ValueError(
            "Expected 199 volumes in each CSV."
        )

    if set(x4_names) != set(x8_names):
        raise ValueError(
            "The 4x and 8x volume names do not match."
        )


def print_statistics(label, rows):
    nmse = values(
        rows,
        "nmse_relative_reduction_percent",
    )
    ssim = values(
        rows,
        "ssim_absolute_improvement",
    )
    psnr = values(
        rows,
        "psnr_improvement_db",
    )

    print(f"\n{label}")
    print("-" * len(label))
    print(
        f"Volumes improved in NMSE: "
        f"{np.sum(nmse > 0)}/{len(nmse)}"
    )
    print(
        f"Volumes improved in SSIM: "
        f"{np.sum(ssim > 0)}/{len(ssim)}"
    )
    print(
        f"Volumes improved in PSNR: "
        f"{np.sum(psnr > 0)}/{len(psnr)}"
    )
    print(
        f"Median NMSE reduction: "
        f"{np.median(nmse):.2f}%"
    )
    print(
        f"Median SSIM improvement: "
        f"{np.median(ssim):.5f}"
    )
    print(
        f"Median PSNR improvement: "
        f"{np.median(psnr):.3f} dB"
    )

    for metric in ["nmse", "ssim", "psnr"]:
        zero = values(rows, f"zero_{metric}")
        recon = values(rows, f"recon_{metric}")

        difference = (
            zero - recon
            if metric == "nmse"
            else recon - zero
        )

        statistic, p_value = wilcoxon(
            difference,
            alternative="greater",
        )

        print(
            f"{metric.upper()} Wilcoxon: "
            f"W={statistic:.0f}, "
            f"p={p_value:.3e}"
        )


def plot_distributions(x4_rows, x8_rows, output):
    panels = [
        (
            "nmse_relative_reduction_percent",
            "NMSE reduction",
            "%",
        ),
        (
            "ssim_absolute_improvement",
            "SSIM improvement",
            "Delta SSIM",
        ),
        (
            "psnr_improvement_db",
            "PSNR improvement",
            "dB",
        ),
    ]

    blue = "#2E4C8C"
    red = "#FA2D1A"

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(15, 5),
        dpi=200,
    )

    for axis, (column, title, ylabel) in zip(
        axes,
        panels,
    ):
        x4 = values(x4_rows, column)
        x8 = values(x8_rows, column)

        boxes = axis.boxplot(
            [x4, x8],
            labels=["4x", "8x"],
            patch_artist=True,
            showfliers=True,
        )

        boxes["boxes"][0].set_facecolor(blue)
        boxes["boxes"][1].set_facecolor(red)

        for box in boxes["boxes"]:
            box.set_alpha(0.85)

        axis.set_title(title, fontweight="bold")
        axis.set_ylabel(ylabel)
        axis.grid(
            axis="y",
            alpha=0.3,
        )

        axis.text(
            1,
            max(x4) * 1.03,
            f"Median {np.median(x4):.3g}",
            ha="center",
            color=blue,
            fontweight="bold",
        )
        axis.text(
            2,
            max(x8) * 1.03,
            f"Median {np.median(x8):.3g}",
            ha="center",
            color=red,
            fontweight="bold",
        )

    figure.suptitle(
        "ReconFormer improves all 199 validation "
        "volumes at both acceleration factors",
        fontsize=17,
        fontweight="bold",
        color=blue,
    )

    figure.tight_layout()
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    figure.savefig(
        output,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze paired 4x and 8x "
            "volume-level reconstruction results."
        )
    )
    parser.add_argument(
        "--x4",
        type=Path,
        default=Path(
            "results/x4_per_volume_metrics.csv"
        ),
    )
    parser.add_argument(
        "--x8",
        type=Path,
        default=Path(
            "results/x8_per_volume_metrics.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/figures/"
            "reconformer_volume_level_consistency.png"
        ),
    )
    args = parser.parse_args()

    x4_rows = read_metrics(args.x4)
    x8_rows = read_metrics(args.x8)

    check_volumes(x4_rows, x8_rows)
    print_statistics("4x acceleration", x4_rows)
    print_statistics("8x acceleration", x8_rows)

    plot_distributions(
        x4_rows,
        x8_rows,
        args.output,
    )

    print(f"\nFigure saved to: {args.output}")


if __name__ == "__main__":
    main()
