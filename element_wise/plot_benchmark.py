from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


Record = dict[str, float | int]


def load_records(path: Path) -> dict[str, list[Record]]:
    required = {
        "method",
        "elements",
        "mean_ms",
        "effective_bandwidth_gbps",
        "bandwidth_utilization_pct",
    }
    grouped: dict[str, list[Record]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"Expected CSV columns {sorted(required)}, got {reader.fieldnames}"
            )
        for row in reader:
            grouped[row["method"]].append(
                {
                    "elements": int(row["elements"]),
                    "mean_ms": float(row["mean_ms"]),
                    "effective_bandwidth_gbps": float(
                        row["effective_bandwidth_gbps"]
                    ),
                    "bandwidth_utilization_pct": float(
                        row["bandwidth_utilization_pct"]
                    ),
                }
            )

    for values in grouped.values():
        values.sort(key=lambda item: int(item["elements"]))
    return dict(grouped)


def plot_metric(axis, records, field: str, title: str, ylabel: str) -> None:
    for method, values in records.items():
        axis.plot(
            [value["elements"] for value in values],
            [value[field] for value in values],
            marker="o",
            label=method,
        )
    axis.set_title(title)
    axis.set_xlabel("Elements")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")


def draw(records: dict[str, list[Record]], output: Path) -> None:
    gpu = {name: values for name, values in records.items() if name.startswith("GPU")}
    cpu = {name: values for name, values in records.items() if name.startswith("CPU")}
    if not gpu:
        raise ValueError("CSV contains no GPU benchmark records")

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    plot_metric(axes[0, 0], gpu, "mean_ms", "GPU execution time", "Mean time (ms)")
    plot_metric(
        axes[0, 1], gpu, "effective_bandwidth_gbps",
        "GPU effective bandwidth", "Effective bandwidth (GB/s)"
    )
    plot_metric(
        axes[1, 0], gpu, "bandwidth_utilization_pct",
        "GPU peak-bandwidth utilization", "Utilization (%)"
    )
    axes[1, 0].axhline(
        100.0, color="black", linestyle="--", linewidth=1, label="Peak DRAM bandwidth"
    )
    if cpu:
        plot_metric(axes[1, 1], cpu, "mean_ms", "CPU execution time", "Mean time (ms)")
    else:
        axes[1, 1].set_visible(False)

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Saved plot to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot element_wise_bench CSV results."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path(__file__).with_name("element_wise_benchmark.csv"),
        help="CSV emitted by element_wise_bench",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).with_name("element_wise_benchmark.png"),
        help="Output PNG path",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    draw(load_records(args.csv), args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
