"""Render the study-overview and saved-model resource figures."""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, ScalarFormatter
import numpy as np


HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
DATA_DIR = Path(os.environ.get("NV_RESULTS_DIR", HERE)).expanduser()
OUTPUT_ROOT = Path(os.environ.get("NV_OUTPUT_DIR", CODE_ROOT / "outputs")).expanduser()
DEFAULT_OUTPUT = OUTPUT_ROOT / "figures"
PREVIEW_OUTPUT = OUTPUT_ROOT / "previews"
INK = "#25282B"
MUTED = "#666C70"
RULE = "#CBD0D3"
GRID = "#E3E7E9"
PALETTE = {
    ("cnn", "torch"): "#285761",
    ("cnn", "numpy"): "#5F8095",
    ("cnn", "onnx"): "#96B0C0",
    ("mlp8", "torch"): "#A64F43",
    ("mlp8", "numpy"): "#D89564",
    ("mlp8", "onnx"): "#8E685B",
}
BACKEND_NAMES = {"torch": "PyTorch", "numpy": "NumPy", "onnx": "ONNX Runtime"}
MODEL_NAMES = {"cnn": "CNN", "mlp8": "MLP8"}

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.7,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    }
)


def save_pair(fig: plt.Figure, output: Path, stem: str) -> None:
    """Export PDF and PNG at the declared physical canvas size."""
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{stem}.pdf", metadata={"Creator": "Matplotlib"})
    fig.savefig(output / f"{stem}.png", dpi=600)
    plt.close(fig)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.75,
            color=MUTED,
            shrinkA=0,
            shrinkB=0,
        )
    )


def label(ax: plt.Axes, x: float, y: float, text: str, *, bold: bool = False) -> None:
    ax.text(
        x,
        y,
        text,
        ha="left",
        va="center",
        fontsize=10.4 if bold else 10.0,
        fontweight="bold" if bold else "normal",
        color=INK if bold else MUTED,
    )


def draw_overview(output: Path) -> None:
    # The four analyses span two NV datasets; resource measurements use
    # separate saved reference models.
    fig = plt.figure(figsize=(183 / 25.4, 121 / 25.4), facecolor="white")
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    label(ax, 0.025, 0.964, "Acquisition-aware accuracy comparisons", bold=True)
    ax.plot([0.025, 0.975], [0.922, 0.922], color=RULE, linewidth=0.65)

    label(ax, 0.025, 0.792, "Weng NV array", bold=True)
    label(ax, 0.025, 0.743, "819 observations · 8 acquisitions")
    label(ax, 0.025, 0.694, "Fixed 24 log-spectral features")
    ax.plot([0.355, 0.408], [0.743, 0.743], color=MUTED, linewidth=0.75)
    ax.plot([0.408, 0.408], [0.630, 0.844], color=MUTED, linewidth=0.75)
    arrow(ax, (0.408, 0.844), (0.451, 0.844))
    arrow(ax, (0.408, 0.630), (0.451, 0.630))

    label(ax, 0.470, 0.853, "a  Training-recipe factorial", bold=True)
    label(ax, 0.470, 0.804, "CNN and MLP8 · 4 scaling conditions")
    label(ax, 0.470, 0.755, "10 seeds · 1,500 / 3,000 epochs")

    label(ax, 0.470, 0.643, "b  Nested compact-model selection", bold=True)
    label(ax, 0.470, 0.594, "12 candidates · inner LOAO, seed 20260804")
    label(ax, 0.470, 0.545, "Outer LOAO · 10 refit seeds")

    label(ax, 0.025, 0.431, "Meneses NV elevator", bold=True)
    label(ax, 0.025, 0.382, "Magnetic time windows · 1D position")
    arrow(ax, (0.388, 0.406), (0.451, 0.406))
    label(ax, 0.470, 0.443, "c  Cross-task scaling comparison", bold=True)
    label(ax, 0.470, 0.394, "CNN and small MLP · 4 scaling conditions")
    label(ax, 0.470, 0.345, "3 seeds · 2 held-out segments")

    label(ax, 0.025, 0.263, "Implementation-aware resource measurements", bold=True)
    ax.plot([0.025, 0.975], [0.222, 0.222], color=RULE, linewidth=0.65)
    label(ax, 0.025, 0.148, "Frozen L31 models", bold=True)
    label(ax, 0.025, 0.098, "First five acquisitions · resource-only")
    arrow(ax, (0.388, 0.123), (0.451, 0.123))
    label(ax, 0.470, 0.173, "d  Multi-backend benchmark", bold=True)
    label(ax, 0.470, 0.124, "PyTorch · NumPy · ONNX Runtime")
    label(ax, 0.470, 0.075, "Batch 1 / 256 · 5 independent processes")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    for text in ax.texts:
        extent = text.get_window_extent(renderer)
        if not canvas.contains(extent.x0, extent.y0) or not canvas.contains(extent.x1, extent.y1):
            raise ValueError(f"Overview label exceeds canvas: {text.get_text()}")
    save_pair(fig, output, "study_l_benchmark_overview")


def resource_rows(source: Path) -> dict[tuple[str, str, int], dict[str, str]]:
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapped = {(row["model"], row["backend"], int(row["batch_size"])): row for row in rows}
    expected = {(model, backend, batch) for model, backend in PALETTE for batch in (1, 256)}
    if len(rows) != 12 or set(mapped) != expected or len(mapped) != len(rows):
        raise ValueError("Expected one observation for all 12 model/backend/batch combinations")
    for row in mapped.values():
        if int(row["process_repeats"]) != 5:
            raise ValueError("Expected five independent processes per implementation")
        values = np.asarray([float(row["rss_after_batch_mib"]), float(row["latency_p50_us"])])
        if not np.isfinite(values).all() or np.any(values <= 0):
            raise ValueError("Resource coordinates must be finite and strictly positive")
    return mapped


def draw_resource(output: Path, source: Path) -> None:
    # Twelve points cover six saved model/backend implementations at two batch sizes.
    # Latency is the median of five process-level p50 values per API call.
    # RSS is post-call process memory, including runtime dependencies.
    data = resource_rows(source)
    fig = plt.figure(figsize=(183 / 25.4, 126 / 25.4), facecolor="white")
    ax = fig.add_axes([0.135, 0.305, 0.825, 0.665])
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set(xlim=(20, 330), ylim=(6.3, 150000))
    ax.set_xlabel("Process RSS (MiB)", fontsize=10.0, labelpad=7)
    ax.set_ylabel("Inference latency (μs per call)", fontsize=10.0, labelpad=7)
    ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 300]))
    ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000, 100000]))
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", labelsize=9.3, length=3.5, pad=4)
    ax.grid(which="major", axis="both", color=GRID, linewidth=0.65)
    ax.set_axisbelow(True)

    implementations = []
    for (model, backend), color in PALETTE.items():
        coords = [
            (
                float(data[(model, backend, batch)]["rss_after_batch_mib"]),
                float(data[(model, backend, batch)]["latency_p50_us"]),
            )
            for batch in (1, 256)
        ]
        ax.plot(*zip(*coords), color=color, alpha=0.35, linewidth=0.75, zorder=2)
        for batch, (rss, latency) in zip((1, 256), coords):
            ax.scatter(
                rss,
                latency,
                s=43,
                marker="o" if batch == 1 else "s",
                facecolor=color,
                edgecolor=INK,
                linewidth=0.65,
                zorder=3,
            )
        implementations.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                markersize=6.2,
                markerfacecolor=color,
                markeredgecolor=INK,
                markeredgewidth=0.65,
                label=f"{MODEL_NAMES[model]} · {BACKEND_NAMES[backend]}",
            )
        )
    # Arrange each backend column with CNN above MLP8.
    implementation_order = [0, 3, 1, 4, 2, 5]
    fig.legend(
        handles=[implementations[index] for index in implementation_order],
        loc="upper center",
        bbox_to_anchor=(0.55, 0.198),
        ncol=3,
        fontsize=9.2,
        handletextpad=0.6,
        columnspacing=1.6,
        labelspacing=0.9,
        borderaxespad=0,
    )
    shapes = [
        Line2D([], [], marker=marker, linestyle="none", markersize=6.2,
               markerfacecolor="white", markeredgecolor=INK, markeredgewidth=0.65,
               label=f"Batch {batch}")
        for batch, marker in ((1, "o"), (256, "s"))
    ]
    fig.legend(
        handles=shapes,
        loc="upper center",
        bbox_to_anchor=(0.55, 0.081),
        ncol=2,
        fontsize=9.2,
        handletextpad=0.6,
        columnspacing=2.0,
        borderaxespad=0,
    )
    save_pair(fig, output, "study_l_resource_tradeoff")
    print("Resource mapping: model, backend, batch -> post-batch RSS MiB, median process p50 μs/call")
    for key, row in data.items():
        print(f"  {key}: {float(row['rss_after_batch_mib']):.6f}, {float(row['latency_p50_us']):.6f}")


def draw_resource_preview(source: Path, *, export_output: Path | None = None) -> None:
    """Plot latency-RSS trade-offs for the saved model/backend combinations."""
    data = resource_rows(source)
    output = export_output if export_output is not None else PREVIEW_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    colors = {
        ("cnn", "torch"): "#1293BC",
        ("cnn", "numpy"): "#26A49A",
        ("cnn", "onnx"): "#617EAE",
        ("mlp8", "torch"): "#F21447",
        ("mlp8", "numpy"): "#E99C23",
        ("mlp8", "onnx"): "#B96082",
    }
    style = {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial"],
        "font.size": 9,
        "text.color": "#222222",
        "axes.labelcolor": "#222222",
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    }
    with mpl.rc_context(style):
        fig = plt.figure(figsize=(150 / 25.4, 110 / 25.4), facecolor="white")
        ax = fig.add_axes([0.157, 0.143, 0.816, 0.616])
        ax.set_xscale("linear")
        ax.set_yscale("log")
        ax.set(xlim=(20, 325), ylim=(8, 110000))
        ax.set_xlabel("Process RSS (MiB)", fontsize=10, labelpad=5)
        ax.set_ylabel("Inference latency (μs per call)", fontsize=10, labelpad=6)
        ax.xaxis.set_major_locator(FixedLocator([50, 100, 200, 300]))
        ax.yaxis.set_major_locator(FixedLocator([10, 100, 1000, 10000, 100000]))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.tick_params(axis="both", which="major", direction="out",
                       labelsize=9, length=3, width=0.8, pad=3)
        ax.grid(which="major", axis="both", color="#E0E0E0", linewidth=0.45)
        ax.set_axisbelow(True)

        handles = []
        plotted = {}
        for (model, backend), color in colors.items():
            coords = np.asarray([
                [float(data[(model, backend, batch)]["rss_after_batch_mib"]),
                 float(data[(model, backend, batch)]["latency_p50_us"])]
                for batch in (1, 256)
            ])
            ax.plot(coords[:, 0], coords[:, 1], color=color,
                    alpha=0.35, linewidth=0.65, zorder=2)
            for batch, (rss, latency) in zip((1, 256), coords):
                points = ax.scatter(rss, latency, s=48,
                                    marker="o" if batch == 1 else "s",
                                    facecolor="white", edgecolor=color,
                                    linewidth=1.3, zorder=3)
                actual = tuple(np.asarray(points.get_offsets())[0])
                expected = (float(data[(model, backend, batch)]["rss_after_batch_mib"]),
                            float(data[(model, backend, batch)]["latency_p50_us"]))
                if actual != expected:
                    raise ValueError("A plotted resource coordinate differs from the source")
                plotted[(model, backend, batch)] = actual
            handles.append(Line2D(
                [], [], marker="o", linestyle="none", markersize=5.8,
                markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=1.3,
                label=f"{MODEL_NAMES[model]} {BACKEND_NAMES[backend]}",
            ))
        if len(plotted) != 12:
            raise ValueError("The resource preview must contain exactly 12 points")
        fig.legend(
            handles=[handles[index] for index in (0, 3, 1, 4, 2, 5)],
            loc="upper center", bbox_to_anchor=(0.565, 0.98), ncol=3,
            fontsize=8.3, handletextpad=0.5, handlelength=1,
            columnspacing=1.3, labelspacing=0.7, borderaxespad=0,
        )
        shapes = [
            Line2D([], [], marker=marker, linestyle="none", markersize=5.8,
                   markerfacecolor="white", markeredgecolor="#555555",
                   markeredgewidth=1.3, label=f"Batch {batch}")
            for batch, marker in ((1, "o"), (256, "s"))
        ]
        fig.legend(handles=shapes, loc="upper center", bbox_to_anchor=(0.565, 0.85),
                   ncol=2, fontsize=8.3, handletextpad=0.5, handlelength=1,
                   columnspacing=1.6, borderaxespad=0)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        for artist in [*ax.get_xticklabels(), *ax.get_yticklabels(),
                       ax.xaxis.label, ax.yaxis.label,
                       *(text for legend in fig.legends for text in legend.get_texts())]:
            bounds = artist.get_window_extent(renderer)
            if not fig.bbox.contains(bounds.x0, bounds.y0) or not fig.bbox.contains(bounds.x1, bounds.y1):
                raise ValueError(f"A resource label exceeds the preview canvas: {artist.get_text()}")
        fig.savefig(output / "study_l_resource_tradeoff.png", dpi=450)
        if export_output is not None:
            fig.savefig(output / "study_l_resource_tradeoff.pdf", metadata={"Creator": "Matplotlib"})
        plt.close(fig)
    mode = "Approved PNG/PDF export" if export_output is not None else "PNG preview"
    print(f"{mode}: 12 unmodified coordinates; six implementation pairs; no new statistics.")
    for key, coordinate in plotted.items():
        print(key, coordinate)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source", type=Path, default=DATA_DIR / "onnx_benchmark_summary.csv")
    parser.add_argument("--resource-only", action="store_true",
                        help="Update the resource figure while preserving the existing overview")
    approved_mode = parser.add_mutually_exclusive_group()
    approved_mode.add_argument("--preview", action="store_true",
                               help="Write the resource PNG to NV_OUTPUT_DIR/previews")
    approved_mode.add_argument("--approved-export", action="store_true",
                               help="Export the resource PNG/PDF files to --output")
    args = parser.parse_args()
    if args.preview:
        draw_resource_preview(args.source)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    draw_resource_preview(args.source, export_output=args.output)


if __name__ == "__main__":
    main()
