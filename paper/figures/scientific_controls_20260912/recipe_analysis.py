"""Audit and plot the fixed ten-seed Weng preprocessing factorial.

Ten optimisation seeds are averaged within each acquisition before the eight-
acquisition summaries and paired bootstrap intervals. Epochs 1500 and 3000 are
checkpoints from the same training trajectory, not independent runs. This
analysis selects neither a model nor a recipe.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE / "recipe_full.csv"
NEW = HERE / "recipe_additional.csv"
SUMMARY = HERE / "recipe_summary.csv"
FIG_ACCURACY = HERE / "recipe_accuracy"
FIG_GAP = HERE / "recipe_gap"

FOLDS = tuple(f"A{i}" for i in range(1, 9))
SEEDS = tuple(range(20260804, 20260814))
EPOCHS = (1500, 3000)
GROUPS = ((0, 0, "Neither scaled"), (0, 1, "Target only"),
          (1, 0, "Input only"), (1, 1, "Both scaled"))
BLUE, RED = "#285761", "#B55847"
INK, MUTED, PALE, RULE = "#202326", "#626A70", "#D4D9DC", "#E8ECEE"
RNG_SEED = 20260930
BOOTSTRAPS = 100_000


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_and_audit():
    """Return 131,040 prediction rows and 1,280 per-fit mean errors in micrometres."""
    old, new = pd.read_csv(OLD), pd.read_csv(NEW)
    if len(old) != 39_312 or len(new) != 91_728:
        raise ValueError("Three- or seven-seed row count changed")
    data = pd.concat((old, new), ignore_index=True)
    if set(data.seed) != set(SEEDS) or set(data.fold) != set(FOLDS):
        raise ValueError("Acquisition or seed inventory changed")
    if len(data) != 131_040 or data.job_id.nunique() != 640:
        raise ValueError("Ten-seed experiment coverage changed")
    if data.data_sha256.nunique() != 1 or data.source_sha256.nunique() != 2:
        raise ValueError("Data or two-stage source fingerprint changed")
    numeric = data[["true_x_um", "true_y_um", "pred_x_um", "pred_y_um", "error_um"]].to_numpy()
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite observation or prediction")
    keys = ["fold", "model", "seed", "input_standardised",
            "target_standardised", "epoch", "row_index"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate observation")
    if (data.groupby(["fold", "seed", "epoch"]).batch_order_sha256.nunique() != 1).any():
        raise ValueError("Batch order differs within a matched fold/seed/epoch")
    if set(data.epoch) != set(EPOCHS):
        raise ValueError("Training budget changed")
    group = data.groupby(keys[:-1], as_index=False).error_um.mean()
    if len(group) != 640 * 2:
        raise ValueError("Per-fit endpoint coverage changed")
    return data, group


def analyse(group):
    """Build eight recipe/checkpoint summaries after averaging seeds by acquisition."""
    per_seed = group.pivot(index=["fold", "seed", "input_standardised",
                                  "target_standardised", "epoch"],
                           columns="model", values="error_um").reset_index()
    if len(per_seed) != 8 * 10 * 4 * 2 or per_seed[["cnn", "mlp8"]].isna().any().any():
        raise ValueError("Matched CNN/MLP8 fit missing")
    per_seed["d_um"] = per_seed.mlp8 - per_seed.cnn
    per_acq = per_seed.groupby(["fold", "input_standardised", "target_standardised", "epoch"],
                               as_index=False)[["cnn", "mlp8", "d_um"]].mean()
    ref = per_acq[(per_acq.input_standardised == 0)
                  & (per_acq.target_standardised == 0)
                  & (per_acq.epoch == 3000)].set_index("fold").d_um
    same_epoch_ref = {
        epoch: per_acq[(per_acq.input_standardised == 0)
                      & (per_acq.target_standardised == 0)
                      & (per_acq.epoch == epoch)].set_index("fold").d_um
        for epoch in EPOCHS
    }
    draws = np.random.default_rng(RNG_SEED).integers(0, 8, size=(BOOTSTRAPS, 8))
    rows = []
    for ix, iy, title in GROUPS:
        for epoch in EPOCHS:
            subset = per_acq[(per_acq.input_standardised == ix)
                             & (per_acq.target_standardised == iy)
                             & (per_acq.epoch == epoch)].sort_values("fold")
            if tuple(subset.fold) != FOLDS:
                raise ValueError("Acquisition order changed")
            values = {name: subset[name].to_numpy() for name in ("cnn", "mlp8", "d_um")}
            values["delta_d_um"] = values["d_um"] - ref.loc[list(FOLDS)].to_numpy()
            # Hold epoch fixed here; delta_d_um retains the original recipe reference.
            values["same_epoch_delta_d_um"] = (
                values["d_um"] - same_epoch_ref[epoch].loc[list(FOLDS)].to_numpy())
            row = {"input_standardised": ix, "target_standardised": iy,
                   "scaling_group": title, "epoch": epoch,
                   "acquisitions": 8, "optimisation_seeds": 10,
                   "mlp8_acquisition_wins": int(np.sum(values["d_um"] < 0))}
            for name, arr in values.items():
                lower, upper = np.quantile(arr[draws].mean(axis=1), [.025, .975])
                row[name + "_mean"] = float(arr.mean())
                row[name + "_acquisition_sd"] = float(arr.std(ddof=1))
                row[name + "_ci_low"] = float(lower)
                row[name + "_ci_high"] = float(upper)
            rows.append(row)
    return rows


def write_summary(rows):
    """Write the eight recipe-checkpoint summaries and source checksums."""
    enriched = [{**row, "old_results_sha256": sha(OLD), "additional_results_sha256": sha(NEW),
                 "bootstrap_seed": RNG_SEED, "bootstrap_draws": BOOTSTRAPS} for row in rows]
    with SUMMARY.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(enriched[0]))
        writer.writeheader()
        writer.writerows(enriched)


def write_factorial_gap_contrasts(group):
    """Write acquisition-level 2x2 contrasts for the MLP8-minus-CNN error gap.

    The retained bootstrap resamples eight acquisitions. Intervals are
    pointwise, and the two trajectory checkpoints remain separate.
    """
    mean = group.groupby(["fold", "model", "input_standardised",
                          "target_standardised", "epoch"], as_index=False).error_um.mean()
    wide = mean.pivot(index=["fold", "input_standardised", "target_standardised", "epoch"],
                      columns="model", values="error_um").reset_index()
    wide["gap"] = wide.mlp8 - wide.cnn
    weights = {
        "Input": [-.5, -.5, .5, .5],
        "Target": [-.5, .5, -.5, .5],
        "Interaction": [1., -1., -1., 1.],
        "Joint": [-1., 0., 0., 1.],
        "Input_given_target_off": [-1., 0., 1., 0.],
        "Input_given_target_on": [0., -1., 0., 1.],
        "Target_given_input_off": [-1., 1., 0., 0.],
        "Target_given_input_on": [0., 0., -1., 1.],
    }
    draws = np.random.default_rng(RNG_SEED).integers(0, 8, size=(BOOTSTRAPS, 8))
    draw_hash = hashlib.sha256(draws.astype(np.int64, copy=False).tobytes()).hexdigest()
    rows = []
    for epoch in EPOCHS:
        matrix = np.array([
            [float(wide[(wide.fold == fold) & (wide.epoch == epoch)
                        & (wide.input_standardised == a)
                        & (wide.target_standardised == b)].gap.iloc[0])
             for a, b in ((0, 0), (0, 1), (1, 0), (1, 1))]
            for fold in FOLDS
        ])
        d00, d01, d10, d11 = matrix.T
        direct = {
            "Input": .5 * ((d10 - d00) + (d11 - d01)),
            "Target": .5 * ((d01 - d00) + (d11 - d10)),
            "Interaction": d11 - d10 - d01 + d00,
            "Joint": d11 - d00,
            "Input_given_target_off": d10 - d00,
            "Input_given_target_on": d11 - d01,
            "Target_given_input_off": d01 - d00,
            "Target_given_input_on": d11 - d10,
        }
        assert np.allclose(direct["Input"] + direct["Target"], direct["Joint"], rtol=0, atol=1e-10)
        for contrast, vector in weights.items():
            values = matrix @ np.asarray(vector)
            if not np.allclose(values, direct[contrast], rtol=0, atol=1e-10):
                raise ValueError(f"Direct and weighted contrast disagree: {contrast}, {epoch}")
            common = {"checkpoint": epoch, "contrast": contrast,
                      "weights_d00_d01_d10_d11": ",".join(map(str, vector)),
                      "optimisation_seeds_averaged": 10, "acquisitions": 8,
                      "bootstrap_seed": RNG_SEED, "bootstrap_draws": BOOTSTRAPS,
                      "resample_indices_sha256": draw_hash,
                      "interval_type": "pointwise percentile 95%",
                      "old_results_sha256": sha(OLD), "additional_results_sha256": sha(NEW)}
            for fold, value in zip(FOLDS, values):
                rows.append({"record_type": "acquisition", **common, "acquisition": fold,
                             "estimate_um": float(value), "acquisition_sd_um": "",
                             "ci_low_um": "", "ci_high_um": ""})
            low, high = np.quantile(values[draws].mean(axis=1), [.025, .975])
            rows.append({"record_type": "summary", **common, "acquisition": "",
                         "estimate_um": float(values.mean()),
                         "acquisition_sd_um": float(values.std(ddof=1)),
                         "ci_low_um": float(low), "ci_high_um": float(high)})
    path = HERE / "recipe_factorial_effects.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return [row for row in rows if row["record_type"] == "summary"]


def style():
    mpl.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "font.size": 9, "axes.labelsize": 10,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.linewidth": 0.75, "axes.spines.top": False,
                         "axes.spines.right": False, "xtick.direction": "out",
                         "ytick.direction": "out", "xtick.major.size": 4,
                         "ytick.major.size": 4, "xtick.major.width": .75,
                         "ytick.major.width": .75, "pdf.fonttype": 42})


def axis_recipe(ax):
    positions = np.array([0, .68, 1.83, 2.51, 3.66, 4.34, 5.49, 6.17])
    ax.set_xticks(positions, ["1,500", "3,000"] * 4)
    ax.set_xlim(-.48, 6.65)
    for x in (1.255, 3.085, 4.915):
        ax.axvline(x, color=RULE, linewidth=.75, zorder=0)
    for i, (_, _, label) in enumerate(GROUPS):
        center = (positions[2 * i] + positions[2 * i + 1]) / 2
        ax.text(center, 1.045, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9.2, color=INK)
    ax.set_xlabel("Training budget (epochs)", labelpad=10)
    return positions


def save(fig, base):
    fig.savefig(base.with_suffix(".png"), dpi=360, facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def plot_accuracy(rows):
    style()
    fig, ax = plt.subplots(figsize=(7.3, 3.75), constrained_layout=False)
    fig.subplots_adjust(left=.105, right=.97, top=.80, bottom=.24)
    xs = axis_recipe(ax)
    for model, color, marker, offset, label in (
        ("cnn", BLUE, "o", -.115, "CNN"),
        ("mlp8", RED, "s", .115, "MLP8"),
    ):
        y = np.array([row[model + "_mean"] for row in rows])
        lower = np.array([row[model + "_ci_low"] for row in rows])
        upper = np.array([row[model + "_ci_high"] for row in rows])
        ax.errorbar(xs + offset, y, yerr=[y - lower, upper - y],
                    fmt=marker, markersize=6.0, color=color, markerfacecolor=color,
                    markeredgecolor=INK, markeredgewidth=.6, elinewidth=1.05,
                    capsize=2.5, capthick=1.0, linestyle="none", label=label, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(30, 190)
    ax.set_yticks([30, 40, 60, 100, 160], ["30", "40", "60", "100", "160"])
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_ylabel("LOAO error (μm; log scale)", labelpad=8)
    ax.legend(loc="upper center", bbox_to_anchor=(.5, 1.26), ncol=2,
              frameon=False, columnspacing=2.0, handletextpad=.5, fontsize=9.2)
    save(fig, FIG_ACCURACY)


def plot_gap(rows):
    style()
    fig, ax = plt.subplots(figsize=(7.3, 3.75), constrained_layout=False)
    fig.subplots_adjust(left=.105, right=.97, top=.80, bottom=.24)
    xs = axis_recipe(ax)
    ax.axhline(0, color="#8C9499", linewidth=1.0, linestyle=(0, (3, 3)), zorder=0)
    for x, row in zip(xs, rows):
        y = row["d_um_mean"]
        color = RED if y < 0 else BLUE
        ax.errorbar(x, y, yerr=[[y - row["d_um_ci_low"]],
                                [row["d_um_ci_high"] - y]],
                    fmt="o", markersize=6.3, color=color, markerfacecolor=color,
                    markeredgecolor=INK, markeredgewidth=.6, elinewidth=1.1,
                    capsize=2.6, capthick=1.0, zorder=3)
    ax.set_ylim(-23, 128)
    ax.set_yticks([-20, 0, 20, 40, 60, 80, 100, 120])
    ax.set_ylabel("MLP8 − CNN error (μm)", labelpad=8)
    ax.text(.99, 1.23, "Below 0: MLP8 lower error     Above 0: CNN lower error",
            transform=ax.transAxes, ha="right", va="center", fontsize=8.5, color=MUTED)
    save(fig, FIG_GAP)


def main():
    _, group = read_and_audit()
    rows = analyse(group)
    write_summary(rows)
    plot_accuracy(rows)
    plot_gap(rows)
    for row in rows:
        print(f"x{row['input_standardised']} y{row['target_standardised']} e{row['epoch']}: "
              f"CNN={row['cnn_mean']:.3f}, MLP8={row['mlp8_mean']:.3f}, "
              f"d={row['d_um_mean']:.3f} [{row['d_um_ci_low']:.3f}, {row['d_um_ci_high']:.3f}] μm")


if __name__ == "__main__":
    main()
