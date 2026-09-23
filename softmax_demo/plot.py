from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


Record = tuple[int, int, float, float, float]


def load_records(path: Path) -> dict[str, list[Record]]:
    expected = {
        "method",
        "rows",
        "cols",
        "elements",
        "mean_ms",
        "effective_bandwidth_gbps",
        "bandwidth_utilization_pct",
    }
    records: dict[str, list[Record]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"Expected CSV columns {sorted(expected)}, got {reader.fieldnames}"
            )
        for row in reader:
            records[row["method"]].append(
                (
                    int(row["cols"]),
                    int(row["elements"]),
                    float(row["mean_ms"]),
                    float(row["effective_bandwidth_gbps"]),
                    float(row["bandwidth_utilization_pct"]),
                )
            )

    for values in records.values():
        values.sort(key=lambda value: value[0])
    return dict(records)


def draw(records: dict[str, list[Record]], output: Path) -> None:
    if not records:
        raise ValueError("CSV contains no benchmark records")

    figure, axes = plt.subplots(4, 1, figsize=(16, 15), sharex=True)
    panels_gpu = (
        (axes[0], 2, "Kernel mean time", "Time (ms)"),
        (axes[1], 3, "Effective memory bandwidth", "Bandwidth (GB/s)"),
        (axes[2], 4, "Theoretical bandwidth utilization", "Utilization (%)"),
    )
    for axis, value_index, title, ylabel in panels_gpu:
        for method, values in records.items():
            if "CPU" in method:
                continue
            columns = [value[0] for value in values]
            metric = [value[value_index] for value in values]
            axis.plot(columns, metric, marker="o", label=method)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small", ncols=2)
    
    for method, values in records.items():
        if "CPU" not in method:
            continue
        columns = [value[0] for value in values]
        metric = [value[2] for value in values]
        axes[-1].plot(columns, metric, marker="o", label=method)
    axes[-1].set_title("CPU Softmax")
    axes[-1].set_ylabel("Time (ms)")
    axes[-1].grid(True, alpha=0.3)
    axes[-1].legend(fontsize="small", ncols=2)

    axes[-1].set_xlabel("Columns per row (rows are fixed by the benchmark)")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"Saved plot to {output.resolve()}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot softmax_bench CSV output.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("softmax_benchmark.csv"),
        help="CSV emitted by softmax_bench (default: ./softmax_benchmark.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=script_dir / "softmax_benchmark.png",
        help="Output image (default: softmax_demo/softmax_benchmark.png)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    draw(load_records(args.csv), args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
