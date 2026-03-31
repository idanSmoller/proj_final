#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr


# -----------------------------
# Configuration
# -----------------------------
PROFESSION_RENAME = {
    "construction laborer": "construction worker",
}

ACTIVATION_METRICS = {
    "continuous_bias": "Continuous bias",
    "mean_based_bias": "Mean-based bias",
}


# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / std


def safe_corr(a: pd.Series, b: pd.Series, method: str = "pearson") -> float:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 3:
        return np.nan
    if x.iloc[:, 0].nunique() <= 1 or x.iloc[:, 1].nunique() <= 1:
        return np.nan
    return x.iloc[:, 0].corr(x.iloc[:, 1], method=method)

def corr_and_pval(a: pd.Series, b: pd.Series, method: str = "pearson") -> tuple[float, float]:
    x = pd.concat([a, b], axis=1).dropna()
    if len(x) < 3:
        return np.nan, np.nan
    if x.iloc[:, 0].nunique() <= 1 or x.iloc[:, 1].nunique() <= 1:
        return np.nan, np.nan
    if method == "pearson":
        r, p = pearsonr(x.iloc[:, 0], x.iloc[:, 1])
    elif method == "spearman":
        r, p = spearmanr(x.iloc[:, 0], x.iloc[:, 1])
    else:
        raise ValueError(f"Unknown method: {method}")
    return r, p


# -----------------------------
# Data loading / cleaning
# -----------------------------
def load_data(
    bias_path: Path,
    text_path: Path,
    baseline_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bias = pd.read_csv(bias_path)
    text = pd.read_csv(text_path)
    baseline_raw = pd.read_csv(baseline_path)

    # Standardize profession naming
    if "profession" in bias.columns:
        bias["profession"] = bias["profession"].replace(PROFESSION_RENAME)

    if "profession" in text.columns:
        text["profession"] = text["profession"].replace(PROFESSION_RENAME)

    if "occupation" in baseline_raw.columns:
        baseline_raw["occupation"] = baseline_raw["occupation"].replace(PROFESSION_RENAME)
        baseline_raw = baseline_raw.rename(columns={"occupation": "profession"})
    elif "profession" in baseline_raw.columns:
        baseline_raw["profession"] = baseline_raw["profession"].replace(PROFESSION_RENAME)
    else:
        raise ValueError("Baseline CSV must contain either 'occupation' or 'profession' column.")

    if "SD" not in baseline_raw.columns:
        raise ValueError("Baseline CSV must contain an 'SD' column.")

    # Keep only the baseline information you explicitly asked for
    baseline = baseline_raw[["profession", "SD"]].copy()
    baseline["SD"] = pd.to_numeric(baseline["SD"], errors="coerce")
    baseline["SD_z"] = zscore_series(baseline["SD"])

    # Numeric cleanup in bias table
    for col in ["step", "continuous_bias", "mean_based_bias"]:
        if col in bias.columns:
            bias[col] = pd.to_numeric(bias[col], errors="coerce")

    # Numeric cleanup in text table
    if "text_embedding_bias" not in text.columns:
        raise ValueError("Text bias CSV must contain 'text_embedding_bias' column.")
    text["text_embedding_bias"] = pd.to_numeric(text["text_embedding_bias"], errors="coerce")

    required_bias_cols = {"profession", "layer", "step"}
    missing_bias_cols = required_bias_cols - set(bias.columns)
    if missing_bias_cols:
        raise ValueError(f"Bias CSV missing required columns: {sorted(missing_bias_cols)}")

    return bias, text, baseline


# -----------------------------
# Aggregation logic
# -----------------------------
def aggregate_activation_by_profession(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = df.groupby("profession", dropna=False)[metric]

    out = grouped.agg(
        act_mean_signed="mean",
        act_mean_abs=lambda s: np.abs(s).mean(),
        act_std="std",
        act_max_abs=lambda s: np.abs(s).max(),
        act_median="median",
    ).reset_index()

    rms = df.groupby("profession")[metric].apply(
        lambda s: np.sqrt(np.mean(np.square(pd.to_numeric(s, errors="coerce").dropna())))
        if pd.to_numeric(s, errors="coerce").dropna().shape[0] > 0 else np.nan
    ).reset_index(name="act_rms")

    out = out.merge(rms, on="profession", how="left")
    return out


def aggregate_activation_by_profession_layer(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    grouped = df.groupby(["profession", "layer"], dropna=False)[metric]

    out = grouped.agg(
        act_mean_signed="mean",
        act_mean_abs=lambda s: np.abs(s).mean(),
        act_std="std",
        act_max_abs=lambda s: np.abs(s).max(),
        act_median="median",
    ).reset_index()

    rms = df.groupby(["profession", "layer"])[metric].apply(
        lambda s: np.sqrt(np.mean(np.square(pd.to_numeric(s, errors="coerce").dropna())))
        if pd.to_numeric(s, errors="coerce").dropna().shape[0] > 0 else np.nan
    ).reset_index(name="act_rms")

    out = out.merge(rms, on=["profession", "layer"], how="left")
    return out


# -----------------------------
# Correlation analysis
# -----------------------------
def step_layer_correlations_vs_baseline(
    bias: pd.DataFrame,
    baseline: pd.DataFrame,
    activation_metric: str,
) -> pd.DataFrame:
    merged = bias[["profession", "layer", "step", activation_metric]].merge(
        baseline[["profession", "SD", "SD_z"]],
        on="profession",
        how="left",
    )

    records = []
    for (layer, step), g in merged.groupby(["layer", "step"], dropna=False):
        records.append(
            {
                "layer": layer,
                "step": step,
                "activation_metric": activation_metric,
                "pearson_vs_SD": safe_corr(g[activation_metric], g["SD"], "pearson"),
                "spearman_vs_SD": safe_corr(g[activation_metric], g["SD"], "spearman"),
                "pearson_vs_SD_z": safe_corr(g[activation_metric], g["SD_z"], "pearson"),
                "spearman_vs_SD_z": safe_corr(g[activation_metric], g["SD_z"], "spearman"),
                "n_professions": g[[activation_metric, "SD"]].dropna().shape[0],
            }
        )

    return pd.DataFrame(records)


def summarize_profession_level_correlations(
    merged: pd.DataFrame,
    feature_cols: List[str],
    prefix: str = "",
) -> pd.DataFrame:
    records = []
    for feat in feature_cols:
        records.append(
            {
                "feature": feat,
                "pearson_vs_SD": safe_corr(merged[feat], merged["SD"], "pearson"),
                "spearman_vs_SD": safe_corr(merged[feat], merged["SD"], "spearman"),
                "pearson_vs_SD_z": safe_corr(merged[feat], merged["SD_z"], "pearson"),
                "spearman_vs_SD_z": safe_corr(merged[feat], merged["SD_z"], "spearman"),
                "n_professions": merged[[feat, "SD"]].dropna().shape[0],
                "context": prefix,
            }
        )
    return pd.DataFrame(records)


def layer_level_correlations_vs_baseline(
    merged_layer: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    records = []
    for layer, g in merged_layer.groupby("layer", dropna=False):
        for feat in feature_cols:
            records.append(
                {
                    "layer": layer,
                    "feature": feat,
                    "pearson_vs_SD": safe_corr(g[feat], g["SD"], "pearson"),
                    "spearman_vs_SD": safe_corr(g[feat], g["SD"], "spearman"),
                    "pearson_vs_SD_z": safe_corr(g[feat], g["SD_z"], "pearson"),
                    "spearman_vs_SD_z": safe_corr(g[feat], g["SD_z"], "spearman"),
                    "n_professions": g[[feat, "SD"]].dropna().shape[0],
                }
            )
    return pd.DataFrame(records)


# -----------------------------
# Plotting
# -----------------------------
def plot_heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    annotate: bool = False,
) -> None:
    if df.empty:
        return

    pivot = df.pivot(index="layer", columns="step", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 4.8))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        interpolation="nearest",
        vmin=-1,
        vmax=1,
        cmap="coolwarm_r",
    )

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Diffusion step")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(f"{value_col} (range -1 to 1)")

    if annotate:
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    out_path: Path,
    label_col: str = "profession",
    annotate_top: int = 10,
) -> None:
    plot_df = df[[x_col, y_col, label_col]].dropna().copy()
    if plot_df.empty:
        return

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    ax.scatter(plot_df[x_col], plot_df[y_col], alpha=0.8)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if plot_df[x_col].nunique() > 1 and plot_df[y_col].nunique() > 1:
        m, b = np.polyfit(plot_df[x_col], plot_df[y_col], 1)
        xs = np.linspace(plot_df[x_col].min(), plot_df[x_col].max(), 100)
        ax.plot(xs, m * xs + b, linewidth=1.5)

    zx = zscore_series(plot_df[x_col])
    zy = zscore_series(plot_df[y_col])
    plot_df["extreme_score"] = np.abs(zx) + np.abs(zy)

    for _, row in plot_df.nlargest(min(annotate_top, len(plot_df)), "extreme_score").iterrows():
        ax.annotate(row[label_col], (row[x_col], row[y_col]), fontsize=8, alpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main pipeline
# -----------------------------
def run_analysis(
    bias_path: Path,
    text_path: Path,
    baseline_path: Path,
    output_dir: Path,
) -> None:
    log_path = output_dir / "correlation_stats.log"
    def log(msg):
        with open(log_path, "a") as f:
            f.write(msg + "\n")
    ensure_dir(output_dir)
    plot_dir = output_dir / "plots"
    ensure_dir(plot_dir)

    bias, text, baseline = load_data(bias_path, text_path, baseline_path)

    # Save profession coverage tables
    pd.DataFrame({"profession": sorted(bias["profession"].dropna().unique())}).to_csv(
        output_dir / "bias_professions.csv", index=False
    )
    pd.DataFrame({"profession": sorted(text["profession"].dropna().unique())}).to_csv(
        output_dir / "text_professions.csv", index=False
    )
    pd.DataFrame({"profession": sorted(baseline["profession"].dropna().unique())}).to_csv(
        output_dir / "baseline_professions.csv", index=False
    )

    shared_professions = sorted(
        set(bias["profession"].dropna().unique())
        & set(text["profession"].dropna().unique())
        & set(baseline["profession"].dropna().unique())
    )
    pd.DataFrame({"profession": shared_professions}).to_csv(
        output_dir / "shared_professions.csv", index=False
    )


    # -------------------------\n    # 1) Text bias vs SD baseline\n    # -------------------------
    text_vs_baseline = text.merge(baseline, on="profession", how="inner").copy()
    text_vs_baseline.to_csv(output_dir / "text_vs_baseline_table.csv", index=False)

    pearson_r, pearson_p = corr_and_pval(text_vs_baseline["text_embedding_bias"], text_vs_baseline["SD"], "pearson")
    spearman_r, spearman_p = corr_and_pval(text_vs_baseline["text_embedding_bias"], text_vs_baseline["SD"], "spearman")
    pearson_r_z, pearson_p_z = corr_and_pval(text_vs_baseline["text_embedding_bias"], text_vs_baseline["SD_z"], "pearson")
    spearman_r_z, spearman_p_z = corr_and_pval(text_vs_baseline["text_embedding_bias"], text_vs_baseline["SD_z"], "spearman")

    log(f"Text bias vs SD: Pearson r={pearson_r:.4f}, p={pearson_p:.4g}; Spearman r={spearman_r:.4f}, p={spearman_p:.4g}")
    log(f"Text bias vs SD_z: Pearson r={pearson_r_z:.4f}, p={pearson_p_z:.4g}; Spearman r={spearman_r_z:.4f}, p={spearman_p_z:.4g}")

    text_summary = pd.DataFrame([
        {
            "signal": "text_embedding_bias",
            "pearson_vs_SD": pearson_r,
            "pearson_vs_SD_p": pearson_p,
            "spearman_vs_SD": spearman_r,
            "spearman_vs_SD_p": spearman_p,
            "pearson_vs_SD_z": pearson_r_z,
            "pearson_vs_SD_z_p": pearson_p_z,
            "spearman_vs_SD_z": spearman_r_z,
            "spearman_vs_SD_z_p": spearman_p_z,
            "n_professions": text_vs_baseline[["text_embedding_bias", "SD"]].dropna().shape[0],
        }
    ])
    text_summary.to_csv(output_dir / "text_vs_baseline_summary.csv", index=False)

    plot_scatter(
        text_vs_baseline,
        x_col="text_embedding_bias",
        y_col="SD",
        title="Text embedding bias vs baseline SD",
        out_path=plot_dir / "scatter_text_embedding_bias_vs_SD.png",
    )

    plot_scatter(
        text_vs_baseline,
        x_col="text_embedding_bias",
        y_col="SD_z",
        title="Text embedding bias vs baseline SD z-score",
        out_path=plot_dir / "scatter_text_embedding_bias_vs_SD_z.png",
    )

    # -------------------------
    # 2) Activation step-layer vs SD
    # -------------------------
    all_step_layer_corr = []

    for metric in ACTIVATION_METRICS:
        if metric not in bias.columns:
            continue

        corr_df = step_layer_correlations_vs_baseline(bias, baseline, metric)
        # Compute p-values for each correlation
        pearson_ps = []
        spearman_ps = []
        for _, row in corr_df.iterrows():
            a = bias[(bias["layer"] == row["layer"]) & (bias["step"] == row["step"])]
            merged = a.merge(baseline, on="profession", how="left")
            r, p = corr_and_pval(merged[metric], merged["SD"], "pearson")
            pearson_ps.append(p)
            r, p = corr_and_pval(merged[metric], merged["SD"], "spearman")
            spearman_ps.append(p)
        corr_df["pearson_vs_SD_p"] = pearson_ps
        corr_df["spearman_vs_SD_p"] = spearman_ps
        corr_df.to_csv(output_dir / f"step_layer_correlations_{metric}_vs_SD.csv", index=False)
        all_step_layer_corr.append(corr_df)

        log(f"Step-layer {metric} vs SD: max |Pearson|={corr_df['pearson_vs_SD'].abs().max():.4f}, min p={corr_df['pearson_vs_SD_p'].min():.4g}")
        log(f"Step-layer {metric} vs SD: max |Spearman|={corr_df['spearman_vs_SD'].abs().max():.4f}, min p={corr_df['spearman_vs_SD_p'].min():.4g}")

        plot_heatmap(
            corr_df,
            value_col="pearson_vs_SD",
            title=f"Pearson: {metric} vs baseline SD",
            out_path=plot_dir / f"heatmap_pearson_{metric}_vs_SD.png",
        )
        plot_heatmap(
            corr_df,
            value_col="spearman_vs_SD",
            title=f"Spearman: {metric} vs baseline SD",
            out_path=plot_dir / f"heatmap_spearman_{metric}_vs_SD.png",
        )

    if all_step_layer_corr:
        all_step_layer_corr_df = pd.concat(all_step_layer_corr, ignore_index=True)
    else:
        all_step_layer_corr_df = pd.DataFrame()

    all_step_layer_corr_df.to_csv(output_dir / "step_layer_correlations_all_vs_SD.csv", index=False)

    # -------------------------
    # 3) Profession-level activation aggregates vs SD
    # -------------------------
    feature_cols = [
        "act_mean_signed",
        "act_mean_abs",
        "act_std",
        "act_max_abs",
        "act_median",
        "act_rms",
    ]

    profession_aggregates = []
    profession_summaries = []
    profession_layer_aggregates = []
    layer_summaries = []

    for metric in ACTIVATION_METRICS:
        if metric not in bias.columns:
            continue

        agg_prof = aggregate_activation_by_profession(bias, metric)
        agg_prof["activation_metric"] = metric
        merged_prof = agg_prof.merge(baseline, on="profession", how="inner")
        merged_prof.to_csv(output_dir / f"profession_aggregates_{metric}_vs_SD.csv", index=False)
        profession_aggregates.append(merged_prof)

        # Add p-values to profession-level summary
        prof_summary = []
        for feat in feature_cols:
            pearson_r, pearson_p = corr_and_pval(merged_prof[feat], merged_prof["SD"], "pearson")
            spearman_r, spearman_p = corr_and_pval(merged_prof[feat], merged_prof["SD"], "spearman")
            pearson_r_z, pearson_p_z = corr_and_pval(merged_prof[feat], merged_prof["SD_z"], "pearson")
            spearman_r_z, spearman_p_z = corr_and_pval(merged_prof[feat], merged_prof["SD_z"], "spearman")
            prof_summary.append({
                "feature": feat,
                "pearson_vs_SD": pearson_r,
                "pearson_vs_SD_p": pearson_p,
                "spearman_vs_SD": spearman_r,
                "spearman_vs_SD_p": spearman_p,
                "pearson_vs_SD_z": pearson_r_z,
                "pearson_vs_SD_z_p": pearson_p_z,
                "spearman_vs_SD_z": spearman_r_z,
                "spearman_vs_SD_z_p": spearman_p_z,
                "n_professions": merged_prof[[feat, "SD"]].dropna().shape[0],
                "context": metric,
                "activation_metric": metric,
            })
            log(f"{metric} {feat} vs SD: Pearson r={pearson_r:.4f}, p={pearson_p:.4g}; Spearman r={spearman_r:.4f}, p={spearman_p:.4g}")
        prof_summary = pd.DataFrame(prof_summary)
        prof_summary.to_csv(output_dir / f"profession_level_correlations_{metric}_vs_SD.csv", index=False)
        profession_summaries.append(prof_summary)

        for feature in ["act_mean_signed", "act_mean_abs", "act_rms"]:
            plot_scatter(
                merged_prof,
                x_col=feature,
                y_col="SD",
                title=f"{metric}: {feature} vs baseline SD",
                out_path=plot_dir / f"scatter_{metric}_{feature}_vs_SD.png",
            )

        agg_prof_layer = aggregate_activation_by_profession_layer(bias, metric)
        agg_prof_layer["activation_metric"] = metric
        merged_prof_layer = agg_prof_layer.merge(baseline, on="profession", how="inner")
        merged_prof_layer.to_csv(output_dir / f"profession_layer_aggregates_{metric}_vs_SD.csv", index=False)
        profession_layer_aggregates.append(merged_prof_layer)

        # Add p-values to layer-level summary
        layer_summary = []
        for layer, g in merged_prof_layer.groupby("layer", dropna=False):
            for feat in feature_cols:
                pearson_r, pearson_p = corr_and_pval(g[feat], g["SD"], "pearson")
                spearman_r, spearman_p = corr_and_pval(g[feat], g["SD"], "spearman")
                pearson_r_z, pearson_p_z = corr_and_pval(g[feat], g["SD_z"], "pearson")
                spearman_r_z, spearman_p_z = corr_and_pval(g[feat], g["SD_z"], "spearman")
                layer_summary.append({
                    "layer": layer,
                    "feature": feat,
                    "pearson_vs_SD": pearson_r,
                    "pearson_vs_SD_p": pearson_p,
                    "spearman_vs_SD": spearman_r,
                    "spearman_vs_SD_p": spearman_p,
                    "pearson_vs_SD_z": pearson_r_z,
                    "pearson_vs_SD_z_p": pearson_p_z,
                    "spearman_vs_SD_z": spearman_r_z,
                    "spearman_vs_SD_z_p": spearman_p_z,
                    "n_professions": g[[feat, "SD"]].dropna().shape[0],
                    "activation_metric": metric,
                })
                log(f"{metric} {feat} layer {layer} vs SD: Pearson r={pearson_r:.4f}, p={pearson_p:.4g}; Spearman r={spearman_r:.4f}, p={spearman_p:.4g}")
        layer_summary = pd.DataFrame(layer_summary)
        layer_summary.to_csv(output_dir / f"layer_level_correlations_{metric}_vs_SD.csv", index=False)
        layer_summaries.append(layer_summary)

        for layer in merged_prof_layer["layer"].dropna().unique():
            sub = merged_prof_layer[merged_prof_layer["layer"] == layer].copy()
            plot_scatter(
                sub,
                x_col="act_mean_abs",
                y_col="SD",
                title=f"{metric} | {layer}: act_mean_abs vs baseline SD",
                out_path=plot_dir / f"scatter_{metric}_{layer}_act_mean_abs_vs_SD.png",
            )

    if profession_aggregates:
        pd.concat(profession_aggregates, ignore_index=True).to_csv(
            output_dir / "profession_aggregates_all_vs_SD.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "profession_aggregates_all_vs_SD.csv", index=False)

    if profession_summaries:
        pd.concat(profession_summaries, ignore_index=True).to_csv(
            output_dir / "profession_level_correlations_all_vs_SD.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "profession_level_correlations_all_vs_SD.csv", index=False)

    if profession_layer_aggregates:
        pd.concat(profession_layer_aggregates, ignore_index=True).to_csv(
            output_dir / "profession_layer_aggregates_all_vs_SD.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "profession_layer_aggregates_all_vs_SD.csv", index=False)

    if layer_summaries:
        pd.concat(layer_summaries, ignore_index=True).to_csv(
            output_dir / "layer_level_correlations_all_vs_SD.csv", index=False
        )
    else:
        pd.DataFrame().to_csv(output_dir / "layer_level_correlations_all_vs_SD.csv", index=False)

    # -------------------------
    # 4) Best step-layer matches vs SD
    # -------------------------
    summary_records = []
    if not all_step_layer_corr_df.empty:
        for metric in all_step_layer_corr_df["activation_metric"].dropna().unique():
            sub = all_step_layer_corr_df[
                all_step_layer_corr_df["activation_metric"] == metric
            ].copy()

            if not sub.empty and sub["pearson_vs_SD"].notna().any():
                best_pearson = sub.loc[sub["pearson_vs_SD"].abs().idxmax()]
            else:
                best_pearson = None

            if not sub.empty and sub["spearman_vs_SD"].notna().any():
                best_spearman = sub.loc[sub["spearman_vs_SD"].abs().idxmax()]
            else:
                best_spearman = None

            summary_records.append(
                {
                    "activation_metric": metric,
                    "best_pearson_layer": None if best_pearson is None else best_pearson["layer"],
                    "best_pearson_step": None if best_pearson is None else best_pearson["step"],
                    "best_pearson_value": None if best_pearson is None else best_pearson["pearson_vs_SD"],
                    "best_spearman_layer": None if best_spearman is None else best_spearman["layer"],
                    "best_spearman_step": None if best_spearman is None else best_spearman["step"],
                    "best_spearman_value": None if best_spearman is None else best_spearman["spearman_vs_SD"],
                }
            )

    pd.DataFrame(summary_records).to_csv(output_dir / "best_step_layer_matches_vs_SD.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare text bias and activation bias against the baseline profession SD column only."
    )
    parser.add_argument("--bias", type=Path, default=Path("/mnt/data/bias_analysis.csv"))
    parser.add_argument("--text", type=Path, default=Path("/mnt/data/text_bias_analysis.csv"))
    parser.add_argument("--baseline", type=Path, default=Path("/mnt/data/occupations_from_chart_approx.csv"))
    parser.add_argument("--outdir", type=Path, default=Path("/mnt/data/bias_vs_sd_outputs"))
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.bias, args.text, args.baseline, args.outdir)