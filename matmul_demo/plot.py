from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# 每行数据: (M, N, K, mean_ms, gflops, ai, bw_gbps, util_pct)
Record = tuple[int, int, int, float, float, float, float, float]

CSV_COLUMNS = {
    "method",
    "M",
    "N",
    "K",
    "mean_ms",
    "gflops",
    "arithmetic_intensity_flop_per_byte",
    "effective_bandwidth_gbps",
    "bandwidth_utilization_pct",
    "device_peak_bandwidth_gbps",
}

SIZE_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def load_records(path: Path) -> tuple[dict[str, list[Record]], float]:
    """按方法分组加载 CSV，并返回设备的理论峰值带宽（绘制 Roofline 上限用）。"""
    records: dict[str, list[Record]] = defaultdict(list)
    peak_gbps = 0.0
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ValueError(
                f"Expected CSV columns {sorted(CSV_COLUMNS)}, got {reader.fieldnames}"
            )
        for row in reader:
            records[row["method"]].append(
                (
                    int(row["M"]),
                    int(row["N"]),
                    int(row["K"]),
                    float(row["mean_ms"]),
                    float(row["gflops"]),
                    float(row["arithmetic_intensity_flop_per_byte"]),
                    float(row["effective_bandwidth_gbps"]),
                    float(row["bandwidth_utilization_pct"]),
                )
            )
            peak_gbps = float(row["device_peak_bandwidth_gbps"])
    if not records:
        raise ValueError("CSV contains no benchmark records")
    for values in records.values():
        values.sort(key=lambda value: value[2])
    return dict(records), peak_gbps


def draw(records: dict[str, list[Record]], output: Path, peak_gbps: float) -> None:
    methods = sorted(records)
    colormap = plt.get_cmap("turbo")
    colors = {
        method: colormap(i / max(len(methods) - 1, 1))
        for i, method in enumerate(methods)
    }
    gpu_methods = [method for method in methods if not method.startswith("CPU")]
    sizes = [value[2] for value in records[gpu_methods[0]]]
    x_pos = list(range(len(sizes)))
    x_labels = [f"{size}\u00b3" for size in sizes]

    figure, axes = plt.subplots(4, 1, figsize=(16, 17))

    # ---- 面板 1：kernel 平均耗时（含 CPU 基线），对数轴 ----
    for method in methods:
        values = records[method]
        axes[0].plot(
            range(len(values)),
            [value[3] for value in values],
            marker="o",
            color=colors[method],
            label=method,
        )
    axes[0].set_ylabel("Time (ms)")
    axes[0].set_yscale("log")
    axes[0].set_title("Kernel mean time (5 warm-ups + 20 measured runs, L2 flushed)")

    # ---- 面板 2：有效带宽（读 A + 读 B + 写 C 的下限口径） ----
    for method in gpu_methods:
        axes[1].plot(
            x_pos,
            [value[6] for value in records[method]],
            marker="o",
            color=colors[method],
            label=method,
        )
    axes[1].set_ylabel("Effective bandwidth (GB/s)")
    axes[1].set_title("Effective memory bandwidth (read A + read B + write C)")

    # ---- 面板 3：实测算力 ----
    for method in gpu_methods:
        axes[2].plot(
            x_pos,
            [value[4] for value in records[method]],
            marker="o",
            color=colors[method],
            label=method,
        )
    axes[2].set_ylabel("Achieved throughput (GFLOPS)")
    axes[2].set_yscale("log")
    axes[2].set_title("Achieved compute throughput (2*M*N*K / time)")

    # ---- 面板 4：Roofline ----
    # 带宽上限：可达 GFLOPS = peak_bw(GB/s) * AI(FLOP/Byte)
    all_ai = sorted({value[5] for values in records.values() for value in values})
    max_gflops = max(
        value[4] for method in gpu_methods for value in records[method]
    )
    ai_lo, ai_hi = all_ai[0] / 2.0, all_ai[-1] * 1.5
    ai_grid = [ai_lo * (ai_hi / ai_lo) ** (i / 199.0) for i in range(200)]
    axes[3].plot(
        ai_grid,
        [peak_gbps * ai for ai in ai_grid],
        color="black",
        linewidth=1.5,
        label=f"bandwidth ceiling ({peak_gbps:.0f} GB/s)",
    )
    axes[3].axhline(
        max_gflops,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"best achieved ({max_gflops:.0f} GFLOPS)",
    )
    # 立方矩阵 AI = n/6，尺寸越大越靠右；颜色区分方法，标记区分尺寸
    marker_by_size = {
        size: SIZE_MARKERS[i % len(SIZE_MARKERS)] for i, size in enumerate(sizes)
    }
    for method in gpu_methods:
        for value in records[method]:
            axes[3].scatter(
                value[5],
                value[4],
                marker=marker_by_size[value[2]],
                color=colors[method],
                s=42,
                edgecolors="black",
                linewidths=0.4,
            )
    size_handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="gray",
            markeredgecolor="black",
            markersize=8,
            label=f"n = {size}",
        )
        for size, marker in marker_by_size.items()
    ]
    axes[3].legend(handles=size_handles + axes[3].get_legend_handles_labels()[0],
                   fontsize="small")
    axes[3].set_xscale("log")
    axes[3].set_yscale("log")
    axes[3].set_xlabel("Arithmetic intensity (FLOP/Byte)")
    axes[3].set_ylabel("Achieved throughput (GFLOPS)")
    axes[3].set_title("Roofline: cube matrix AI = n/6, grows with size")
    # 把每个尺寸对应的 AI 直接标在 x 轴上，方便把散点簇映射回矩阵尺寸
    axes[3].set_xticks(
        [value[5] for value in records[gpu_methods[0]]],
        labels=[f"{size}\u00b3" for size in sizes],
        minor=True,
    )

    for axis in axes:
        axis.grid(True, alpha=0.3, which="both")
    for axis in axes[:3]:
        axis.legend(fontsize="small", ncols=3)
        axis.set_xticks(x_pos)
        axis.set_xticklabels([""] * len(x_pos))
    axes[2].set_xticklabels(x_labels)
    axes[2].set_xlabel("Matrix size (M = N = K)")

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    print(f"Saved plot to {output.resolve()}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Plot matmul_bench CSV output.")
    parser.add_argument(
        "csv",
        nargs="?",
        type=Path,
        default=script_dir / "matmul_benchmark.csv",
        help="CSV emitted by matmul_bench (default: ./matmul_benchmark.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=script_dir / "matmul_benchmark.png",
        help="Output image (default: matmul_demo/matmul_benchmark.png)",
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    records, peak_gbps = load_records(args.csv)
    draw(records, args.output, peak_gbps)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
