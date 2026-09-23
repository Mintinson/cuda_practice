from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_records(path: Path) -> dict[tuple[str, str], list[tuple[int, float]]]:
    records: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        expected = {"dtype", "method", "elements", "mean_ms"}
        columns = set(reader.fieldnames or ())
        if columns not in (expected, expected - {"dtype"}):
            raise ValueError(
                f"Expected CSV columns {sorted(expected)} (or legacy CSV without dtype), "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            records[(row.get("dtype") or "float32", row["method"])].append(
                (int(row["elements"]), float(row["mean_ms"]))
            )

    for values in records.values():
        values.sort(key=lambda item: item[0])
    return dict(records)


def draw(records: dict[tuple[str, str], list[tuple[int, float]]], output: Path) -> None:
    panels = []
    for dtype in ("float32", "int32"):
        for device in ("GPU", "CPU"):
            data = {
                method: values
                for (record_dtype, method), values in records.items()
                if record_dtype == dtype and method.startswith(device)
            }
            if data:
                panels.append((f"{dtype} {device} reductions", data))
    if not panels:
        raise ValueError("CSV contains no CPU or GPU benchmark records")

    figure, axes = plt.subplots(
        len(panels), 1, figsize=(12, 5 * len(panels)), squeeze=False
    )
    for axis, (title, data) in zip(axes.flat, panels):
        for method, values in data.items():
            elements = [value[0] for value in values]
            milliseconds = [value[1] for value in values]
            axis.plot(elements, milliseconds, marker="o", label=method)
        axis.set_title(title)
        axis.set_xlabel("Elements")
        axis.set_ylabel("Mean time (ms)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small", ncols=2)

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    print(f"Saved plot to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reduce_bench benchmark CSV output, separated by data type."
    )
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=Path("reduce_benchmark.csv"),
        help="CSV emitted by reduce_bench (default: reduce_benchmark.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("reduce_benchmark.png"),
        help="Output image path (default: reduce_benchmark.png)",
    )
    parser.add_argument(
        "--show", action="store_true", help="Also open an interactive plot window"
    )
    args = parser.parse_args()

    draw(load_records(args.csv), args.output)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
