"""Render nested-acquisition and external-task comparisons from stored predictions.

The script aggregates existing predictions without fitting or resampling models.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "pdf.fonttype": 42, "font.size": 9.5,
    "axes.labelsize": 10, "xtick.labelsize": 9.5, "ytick.labelsize": 9.5,
    "axes.linewidth": .65, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": .65, "ytick.major.width": .65,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[2]
ROOT = Path(os.environ.get("NV_PROJECT_DIR", CODE_ROOT)).expanduser()
DATA_DIR = Path(os.environ.get("NV_RESULTS_DIR", HERE)).expanduser()
OUTPUT_ROOT = Path(os.environ.get("NV_OUTPUT_DIR", CODE_ROOT / "outputs")).expanduser()
FOLDS = tuple(f"A{i}" for i in range(1, 9))
SEEDS = tuple(range(20260804, 20260814))
BLUE, RED = "#285761", "#B55847"
INK, MUTED, RULE = "#202326", "#626A70", "#DADFE0"
WIDTH = 183 / 25.4
DEFAULT_OUT = OUTPUT_ROOT / "figures"
PREVIEW_OUT = OUTPUT_ROOT / "previews"

PREVIEW_BLUE = "#168CBB"
PREVIEW_RED = "#ED1746"
PREVIEW_NEUTRAL = "#646A6D"
RECIPE_COLORS = {
    (0, 0): "#168CBB",
    (0, 1): "#ED1746",
    (1, 0): "#F0A600",
    (1, 1): "#9063B5",
}


def neural_reference_errors():
    columns = ["fold", "model", "seed", "input_standardised",
               "target_standardised", "epoch", "error_um", "row_index"]
    data = pd.concat([pd.read_csv(DATA_DIR / name, usecols=columns)
                      for name in ("recipe_full.csv", "recipe_additional.csv")],
                     ignore_index=True)
    if len(data) != 131040 or set(data.seed) != set(SEEDS):
        raise ValueError("The fixed ten-seed recipe inventory has changed")
    references = {}
    for model, xstd, ystd, epoch in (("mlp8", 1, 1, 1500), ("cnn", 0, 0, 3000)):
        selected = data[(data.model == model) & (data.input_standardised == xstd)
                        & (data.target_standardised == ystd) & (data.epoch == epoch)]
        grouped = selected.groupby(["fold", "seed"]).error_um.mean()
        if len(grouped) != 80 or not np.isfinite(grouped).all():
            raise ValueError("Incomplete reference fold/seed coverage")
        references[model] = grouped.groupby("fold").mean().reindex(FOLDS).to_numpy()
    return references


def nested_data():
    frozen = json.loads((DATA_DIR / "nested_loao.json").read_text(encoding="utf-8"))
    extra = pd.read_csv(DATA_DIR / "nested_outer_additional.csv")
    if set(extra.seed) != set(SEEDS[3:]) or set(extra.outer_fold) != set(FOLDS):
        raise ValueError("Incomplete additional nested outer refits")
    sparse = pd.read_csv(ROOT / "results/study_l34_session_aggregates.csv")
    sparse = sparse[sparse.evaluation == "strict_session_loso"].set_index("session")
    selected, nested, linear = [], [], []
    for fold in FOLDS:
        item = frozen["folds"][fold]
        old = {row["seed"]: row["mean_error_um"] for row in item["outer_outcomes"]}
        new = extra[extra.outer_fold == fold].groupby("seed").error_um.mean().to_dict()
        if set(old) != set(SEEDS[:3]) or set(new) != set(SEEDS[3:]):
            raise ValueError(f"Incomplete nested seed inventory in {fold}")
        if set(extra[extra.outer_fold == fold].selected_candidate) != {
                item["selected_candidate"]["candidate"]}:
            raise ValueError("Additional refit differs from the frozen selection")
        selected.append(item["selected_candidate"]["candidate"].split("_")[0].upper())
        nested.append(np.mean(list(old.values()) + list(new.values())))
        linear.append(float(sparse.loc[item["held_out_session"], "sparse16_error_um"]))
    expected = ["MLP32", "MLP8", "MLP8", "MLP8", "MLP8", "MLP16", "MLP8", "MLP8"]
    if selected != expected:
        raise ValueError("Frozen nested candidate identities have changed")
    references = neural_reference_errors()
    values = np.array([nested, references["mlp8"], references["cnn"], linear])
    if values.shape != (4, 8) or not np.isfinite(values).all():
        raise ValueError("Invalid acquisition-level error table")
    if not np.allclose(values[0, [1, 2, 3, 4, 6, 7]],
                       values[1, [1, 2, 3, 4, 6, 7]], atol=1e-10, rtol=0):
        raise ValueError("Selected MLP8 refits do not match their fixed references")
    return selected, values


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(out / f"{name}.{extension}", dpi=450)
    plt.close(fig)


def plot_nested(out):
    selected, values = nested_data()
    means, spread = values.mean(axis=1), values.std(axis=1, ddof=1)
    fig = plt.figure(figsize=(WIDTH, 83 / 25.4))
    ax = fig.add_axes([.193, .195, .782, .73])
    ax.set_xlim(-.5, 9.5)
    ax.set_ylim(-.05, 5.75)
    ax.axis("off")
    cmap = LinearSegmentedColormap.from_list("error", ["#F7F9F9", "#ABC0C3"])
    norm = Normalize(vmin=30, vmax=105)
    strip = {"MLP8": "#F3E5E1", "MLP16": "#E7C8BF", "MLP32": "#D9A698"}
    for j, fold in enumerate(FOLDS):
        ax.text(j, 5.43, fold, ha="center", va="center", fontsize=10)
        ax.add_patch(Rectangle((j - .49, 4.22), .98, .69,
                               facecolor=strip[selected[j]], edgecolor="white", lw=.65))
        ax.text(j, 4.565, selected[j], ha="center", va="center", fontsize=9)
    ax.text(-.65, 4.565, "Inner selection", ha="right", va="center", fontsize=9.5,
            color=INK, clip_on=False)
    ax.text(8.58, 5.43, "Mean ± SD", ha="center", va="center", fontsize=10)
    ax.text(8.58, 4.565, "n = 8 acquisitions", ha="center", va="center", fontsize=9,
            color=MUTED)
    labels = ("Nested selector", "Fixed MLP8", "CNN", "Sparse16")
    for i, (label, color) in enumerate(zip(labels, (RED, RED, BLUE, MUTED))):
        y = 3.53 - i
        ax.text(-.65, y, label, ha="right", va="center", fontsize=10,
                color=color, clip_on=False)
        for j, value in enumerate(values[i]):
            ax.add_patch(Rectangle((j - .49, y - .43), .98, .86,
                                   facecolor=cmap(norm(value)), edgecolor="white", lw=.6))
            ax.text(j, y, f"{value:.1f}", ha="center", va="center", fontsize=9.4,
                    color=INK)
        ax.text(8.58, y, f"{means[i]:.2f} ± {spread[i]:.2f}", ha="center", va="center",
                fontsize=9.2, color=color)
    ax.plot([7.61, 7.61], [-.01, 3.99], color=RULE, lw=.65)
    fig.text(.193, .947, "Held-out acquisition", fontsize=10, color=INK)
    cax = fig.add_axes([.193, .073, .36, .029])
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                           cax=cax, orientation="horizontal", ticks=[30, 60, 90, 105])
    colorbar.ax.tick_params(length=2.5, labelsize=9, pad=2)
    colorbar.outline.set_linewidth(.5)
    fig.text(.588, .087, "Localisation error (µm)", fontsize=9.5, va="center")
    save(fig, out, "study_l_nested_selection")
    print("Nested acquisition means:")
    for label, row, mean, sd in zip(labels, values, means, spread):
        print(f"{label}: {row.tolist()}; mean={mean:.10f}; sample_sd={sd:.10f}")


def external_data():
    columns = ["model", "seed", "input_standardised", "target_standardised",
               "epoch", "test_segment", "window_index", "abs_error_m",
               "parking_ground_truth"]
    data = pd.read_csv(DATA_DIR / "meneses_full.csv", usecols=columns)
    data = data[(data.epoch == 400) & (data.parking_ground_truth == 1)]
    if set(data.seed) != set(SEEDS[:3]) or set(data.test_segment) != {2, 3}:
        raise ValueError("External-task seed/segment inventory has changed")
    keys = ["input_standardised", "target_standardised", "model", "test_segment", "seed"]
    if data.duplicated(keys + ["window_index"]).any():
        raise ValueError("Duplicate external-task endpoint prediction")
    counts = data.groupby(keys).size()
    for segment, expected in ((2, 2144), (3, 2305)):
        if not counts.xs(segment, level="test_segment").eq(expected).all():
            raise ValueError("Parking-target window counts have changed")
    seed_means = data.groupby(keys).abs_error_m.mean()
    means = seed_means.groupby(keys[:-1]).mean()
    if len(means) != 16 or not np.isfinite(means).all():
        raise ValueError("Incomplete external preprocessing comparison")
    return means


def plot_external(out):
    data = external_data()
    fig = plt.figure(figsize=(WIDTH, 86 / 25.4))
    ax = fig.add_axes([.232, .24, .722, .67])
    ax.set_xscale("log")
    ax.set_xlim(.50, 22)
    ax.set_ylim(-.53, 3.50)
    groups = ((0, 0, "Neither"), (0, 1, "Target only"),
              (1, 0, "Input only"), (1, 1, "Input + target"))
    positions = np.arange(4)[::-1]
    ax.set_yticks(positions, [group[2] for group in groups])
    ax.tick_params(axis="y", length=0, pad=12)
    ax.spines["left"].set_visible(False)
    ax.set_xticks([.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
    ax.set_xticks([], minor=True)
    ax.grid(axis="x", color="#E2E6E6", lw=.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Mean absolute error (m; log scale)", labelpad=9)
    fig.text(.035, .94, "Standardisation", fontsize=10, va="bottom")
    for j, (xstd, ystd, _) in enumerate(groups):
        ybase = positions[j]
        if j < 3:
            ax.axhline(ybase - .50, color="#EFF1F1", lw=.6, zorder=0)
        for model, color, offset in (("cnn", BLUE, .17), ("mlp8", RED, -.17)):
            segment_values = [float(data.loc[(xstd, ystd, model, segment)]) for segment in (2, 3)]
            mean = float(np.mean(segment_values))
            y = ybase + offset
            for value, marker in zip(segment_values, ("^", "v")):
                ax.plot(value, y, marker=marker, ms=4.3, linestyle="none",
                        markerfacecolor="white", markeredgecolor="black",
                        markeredgewidth=.45, zorder=3)
            ax.plot(mean, y, marker="o", ms=6.5, linestyle="none", color=color,
                    markeredgecolor="black", markeredgewidth=.45, zorder=4)
            ax.annotate(f"{mean:.3f}", (mean, y), xytext=(10, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=9.3, color=color, zorder=5)
            print(f"External x{xstd}y{ystd}/{model}: seg2={segment_values[0]:.10f}, "
                  f"seg3={segment_values[1]:.10f}, equal_mean={mean:.10f}")
    model_handles = [Line2D([], [], color=color, marker="o", ms=6, linestyle="none",
                            markeredgecolor="black", markeredgewidth=.45, label=label)
                     for color, label in ((BLUE, "CNN"), (RED, "Small MLP"))]
    fig.legend(handles=model_handles, loc="upper right", bbox_to_anchor=(.953, .992),
               ncol=2, handletextpad=.6, columnspacing=1.8, fontsize=9.5)
    marker_handles = [
        Line2D([], [], color=INK, marker="o", ms=6, linestyle="none",
               markerfacecolor=MUTED, markeredgecolor="black", markeredgewidth=.45,
               label="Equal-segment mean"),
        Line2D([], [], color=INK, marker="^", ms=4.3, linestyle="none",
               markerfacecolor="white", markeredgecolor="black", markeredgewidth=.45,
               label="Segment 2"),
        Line2D([], [], color=INK, marker="v", ms=4.3, linestyle="none",
               markerfacecolor="white", markeredgecolor="black", markeredgewidth=.45,
               label="Segment 3"),
    ]
    fig.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(.59, .012),
               ncol=3, handletextpad=.6, columnspacing=1.5, fontsize=9,
               labelcolor=MUTED)
    save(fig, out, "study_l_external_task")


def preview_style():
    """Configure typography for the manuscript figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 10.2,
        "axes.labelsize": 10.8,
        "axes.titlesize": 11.1,
        "xtick.labelsize": 10.2,
        "ytick.labelsize": 10.2,
        "axes.linewidth": .85,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": .85,
        "ytick.major.width": .85,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def save_preview(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist in fig.findobj(matplotlib.text.Text):
        if artist.get_visible() and artist.get_text():
            box = artist.get_window_extent(renderer)
            assert box.x0 >= -1 and box.y0 >= -1, artist.get_text()
            assert box.x1 <= fig.bbox.width+1 and box.y1 <= fig.bbox.height+1, artist.get_text()
    fig.savefig(out / f"{name}.png", dpi=450, facecolor="white")
    if '--preview' not in sys.argv:
        plt.rcParams['pdf.fonttype'] = 42
        fig.savefig(out / f"{name}.pdf", facecolor='white')
    plt.close(fig)


def plot_nested_preview(out):
    """Plot acquisition errors and the model selected in each outer fold."""
    selected, values = nested_data()
    means = values.mean(axis=1)
    spread = values.std(axis=1, ddof=1)
    labels = ("Nested\nselection", "Fixed\nMLP8", "CNN", "Sparse16")
    colours = (PREVIEW_RED, PREVIEW_RED, PREVIEW_BLUE, PREVIEW_NEUTRAL)
    mean_markers = ("o", "o", "o", "o")
    acquisition_offsets = np.linspace(-.14, .14, len(FOLDS))

    fig, ax = plt.subplots(figsize=(150 / 25.4, 88 / 25.4))
    fig.subplots_adjust(left=.14, right=.975, bottom=.18, top=.80)

    nested_markers = {"MLP8": "o", "MLP16": "^", "MLP32": "D"}
    for method_index, (colour, mean_marker) in enumerate(zip(colours, mean_markers)):
        x = method_index + acquisition_offsets
        if method_index == 0:
            for x_value, y_value, candidate in zip(x, values[method_index], selected):
                ax.scatter(x_value, y_value, s=9, marker=nested_markers[candidate],
                           facecolor=colour, edgecolor="white", linewidth=.45,
                           alpha=.68, zorder=3)
        elif method_index == 1:
            ax.scatter(x, values[method_index], s=9, marker="o",
                       facecolor="white", edgecolor=colour, linewidth=.85,
                       alpha=.82, zorder=3)
        elif method_index == 2:
            ax.scatter(x, values[method_index], s=9, marker="o",
                       facecolor=colour, edgecolor="white", linewidth=.45,
                       alpha=.66, zorder=3)
        else:
            ax.scatter(x, values[method_index], s=9, marker="^",
                       facecolor=colour, edgecolor="white", linewidth=.45,
                       alpha=.60, zorder=3)

        ax.errorbar(method_index, means[method_index], yerr=spread[method_index],
                    fmt=mean_marker, ms=4.8, mfc="white", mec=colour, mew=1.1,
                    ecolor=colour, elinewidth=1.15, capsize=2.0, capthick=1.15,
                    zorder=5)

    ax.set_xlim(-.52, 3.52)
    ax.set_ylim(25, 106)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xticks(np.arange(4), labels)
    ax.set_ylabel("Outer localisation error (µm)")
    ax.tick_params(axis="x", length=0, pad=7)
    ax.spines["bottom"].set_linewidth(.85)
    ax.spines["left"].set_linewidth(.85)

    choice_handles = [
        Line2D([], [], marker="o", linestyle="none", markersize=5.2,
               markerfacecolor=PREVIEW_RED, markeredgecolor="white",
               label="MLP8: A2–A5, A7–A8"),
        Line2D([], [], marker="^", linestyle="none", markersize=5.6,
               markerfacecolor=PREVIEW_RED, markeredgecolor="white",
               label="MLP16: A6"),
        Line2D([], [], marker="D", linestyle="none", markersize=5.0,
               markerfacecolor=PREVIEW_RED, markeredgecolor="white",
               label="MLP32: A1"),
    ]
    ax.legend(handles=choice_handles, title="Inner selection",
              loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=3,
              fontsize=8.0, title_fontsize=9.2, handletextpad=.25,
              columnspacing=.9, borderaxespad=0)
    save_preview(fig, out, "study_l_nested_selection")
    print(f"Preview nested points: {values.size}; summaries: {len(means)}")


def plot_external_preview(out):
    """Plot segment-level and equal-segment errors for both task-specific models."""
    data = external_data()
    groups = ((0, 0, "Neither"), (0, 1, "Target only"),
              (1, 0, "Input only"), (1, 1, "Both"))
    fig = plt.figure(figsize=(150 / 25.4, 99 / 25.4))
    axes = [fig.add_axes([left, .39, .36, .546]) for left in (.13, .615)]

    for panel_index, (ax, model, title) in enumerate(zip(
            axes, ("cnn", "mlp8"), ("CNN", "Small MLP"))):
        ax.set_yscale("log")
        ax.set_xlim(-.55, 3.55)
        ax.set_ylim(.50, 22)
        ax.set_yticks([.5, 1, 2, 5, 10, 20], ["0.5", "1", "2", "5", "10", "20"])
        ax.set_yticks([], minor=True)
        ax.set_xticks(range(4), [group[2] for group in groups],
                      rotation=35, ha='right', rotation_mode='anchor')
        ax.tick_params(pad=3.3)
        left = ax.get_position().x0
        fig.text(left+.18, .095, f'({chr(ord("a")+panel_index)}) {title}',
                 ha='center', va='center', fontsize=10.4)
        if panel_index == 0:
            ax.set_ylabel("Mean absolute error (m)", labelpad=6)

        for x_value, (xstd, ystd, _) in enumerate(groups):
            colour = RECIPE_COLORS[(xstd, ystd)]
            segment_values = [float(data.loc[(xstd, ystd, model, segment)])
                              for segment in (2, 3)]
            equal_mean = float(np.mean(segment_values))
            ax.scatter(x_value-.16, segment_values[0], s=10, marker="^",
                       facecolor=colour, edgecolor="white", linewidth=.45,
                       alpha=.72, zorder=3)
            ax.scatter(x_value+.16, segment_values[1], s=10, marker="v",
                       facecolor=colour, edgecolor="white", linewidth=.45,
                       alpha=.72, zorder=3)
            ax.plot(x_value, equal_mean, marker="o", linestyle='none', ms=4.8,
                    mfc="white", mec=colour, mew=1.1, zorder=4)

    fig.text(.55,.165,'Standardisation',ha='center',va='center',fontsize=10.8)
    marker_handles = [
        Line2D([], [], marker="^", linestyle="none", markersize=4.8,
               markerfacecolor=PREVIEW_NEUTRAL, markeredgecolor="white",
               label="Segment 2"),
        Line2D([], [], marker="v", linestyle="none", markersize=4.8,
               markerfacecolor=PREVIEW_NEUTRAL, markeredgecolor="white",
               label="Segment 3"),
        Line2D([], [], marker="o", linestyle="none", markersize=7.0,
               markerfacecolor="white", markeredgecolor=PREVIEW_NEUTRAL,
               markeredgewidth=1.3, label="Equal-segment mean"),
    ]
    fig.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(.55, .013),
               ncol=3, fontsize=8.3, handletextpad=.4, columnspacing=1.1)

    save_preview(fig, out, "study_l_external_task")
    print("Preview external points: 16 segment means; 8 equal-segment means")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--preview", action="store_true",
                        help="write PNG previews to NV_OUTPUT_DIR/previews")
    parser.add_argument("--approved-export", action="store_true",
                        help="export PNG and vector PDF files to --out-dir")
    args = parser.parse_args()
    if args.preview and args.approved_export:
        parser.error('Choose preview or approved export, not both')
    preview_style()
    out = PREVIEW_OUT if args.preview else args.out_dir
    plot_nested_preview(out)
    plot_external_preview(out)


if __name__ == "__main__":
    main()
