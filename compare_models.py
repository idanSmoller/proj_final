#!/usr/bin/env python3
"""
Cross-Model Bias Comparison

Compares bias patterns across different generative models.
Generates comparative visualizations and statistical summaries.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_bias_results(results_path: str) -> pd.DataFrame:
    """Load bias analysis CSV."""
    return pd.read_csv(results_path)


def load_text_bias_results(text_bias_path: str) -> pd.DataFrame:
    """Load text embedding bias CSV."""
    return pd.read_csv(text_bias_path)


def plot_profession_comparison(
    model_data: Dict[str, pd.DataFrame],
    profession: str,
    output_path: str
):
    """Plot bias comparison for a specific profession across models."""
    fig, axes = plt.subplots(len(model_data), 1, figsize=(14, 4 * len(model_data)))

    if len(model_data) == 1:
        axes = [axes]

    for idx, (model_name, df) in enumerate(model_data.items()):
        prof_data = df[df['profession'] == profession]

        if prof_data.empty:
            axes[idx].text(0.5, 0.5, f'No data for {profession}',
                          ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{model_name}: {profession}')
            continue

        # Reshape data for heatmap
        layers = sorted(prof_data['layer'].unique())
        steps = sorted(prof_data['step'].unique())

        grid = np.zeros((len(layers), len(steps)))
        layer_to_idx = {layer: i for i, layer in enumerate(layers)}

        for _, row in prof_data.iterrows():
            l_idx = layer_to_idx.get(row['layer'])
            s_idx = row['step']
            if l_idx is not None and s_idx < len(steps):
                grid[l_idx, s_idx] = row['continuous_bias']

        max_val = np.nanmax(np.abs(grid))
        if max_val == 0 or np.isnan(max_val):
            max_val = 1

        im = axes[idx].imshow(grid, cmap='coolwarm', origin='lower',
                              aspect='auto', vmin=-max_val, vmax=max_val)

        axes[idx].set_yticks(range(len(layers)))
        axes[idx].set_yticklabels(layers)

        if len(steps) > 10:
            tick_indices = range(0, len(steps), 5)
            axes[idx].set_xticks(tick_indices)
            axes[idx].set_xticklabels([steps[i] for i in tick_indices])
        else:
            axes[idx].set_xticks(range(len(steps)))
            axes[idx].set_xticklabels(steps)

        axes[idx].set_xlabel('Timestep')
        axes[idx].set_ylabel('Layer')
        axes[idx].set_title(f'{model_name}: {profession}')

        plt.colorbar(im, ax=axes[idx], label='Bias')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_text_bias_comparison(
    text_bias_data: Dict[str, pd.DataFrame],
    output_path: str
):
    """Compare text embedding bias across models."""
    # Combine all model data
    all_data = []
    for model_name, df in text_bias_data.items():
        df = df.copy()
        df['model'] = model_name
        all_data.append(df)

    combined = pd.concat(all_data, ignore_index=True)

    # Create pivot table
    pivot = combined.pivot(index='profession', columns='model', values='text_embedding_bias')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Heatmap
    im = ax1.imshow(pivot.values, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(pivot.columns)))
    ax1.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax1.set_yticks(range(len(pivot.index)))
    ax1.set_yticklabels(pivot.index)
    ax1.set_title('Text Embedding Bias by Model and Profession')
    plt.colorbar(im, ax=ax1, label='Bias Score')

    # Bar plot
    pivot.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_xlabel('Profession')
    ax2.set_ylabel('Bias Score')
    ax2.set_title('Text Embedding Bias Comparison')
    ax2.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_aggregate_statistics(
    model_data: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Compute aggregate bias statistics per model."""
    stats = []

    for model_name, df in model_data.items():
        stats.append({
            'model': model_name,
            'mean_bias': df['continuous_bias'].mean(),
            'std_bias': df['continuous_bias'].std(),
            'abs_mean_bias': df['continuous_bias'].abs().mean(),
            'max_bias': df['continuous_bias'].abs().max(),
            'masculine_count': (df['continuous_bias'] > 0.1).sum(),
            'feminine_count': (df['continuous_bias'] < -0.1).sum(),
            'neutral_count': (df['continuous_bias'].abs() <= 0.1).sum(),
        })

    return pd.DataFrame(stats)


def plot_model_summary(
    model_data: Dict[str, pd.DataFrame],
    output_path: str
):
    """Plot summary statistics across models."""
    stats = compute_aggregate_statistics(model_data)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean absolute bias
    axes[0, 0].bar(stats['model'], stats['abs_mean_bias'])
    axes[0, 0].set_ylabel('Mean Absolute Bias')
    axes[0, 0].set_title('Average Bias Magnitude by Model')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Distribution of bias
    axes[0, 1].bar(stats['model'], stats['mean_bias'])
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_ylabel('Mean Bias')
    axes[0, 1].set_title('Average Bias Direction by Model\n(Positive=Masculine, Negative=Feminine)')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Bias count breakdown
    x = np.arange(len(stats['model']))
    width = 0.25
    axes[1, 0].bar(x - width, stats['masculine_count'], width, label='Masculine (>0.1)')
    axes[1, 0].bar(x, stats['neutral_count'], width, label='Neutral (±0.1)')
    axes[1, 0].bar(x + width, stats['feminine_count'], width, label='Feminine (<-0.1)')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(stats['model'], rotation=45)
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Bias Distribution by Model')
    axes[1, 0].legend()

    # Standard deviation
    axes[1, 1].bar(stats['model'], stats['std_bias'])
    axes[1, 1].set_ylabel('Standard Deviation')
    axes[1, 1].set_title('Bias Variability by Model')
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_report(
    model_data: Dict[str, pd.DataFrame],
    text_bias_data: Dict[str, pd.DataFrame],
    output_path: str
):
    """Create text summary report."""
    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("CROSS-MODEL GENDER BIAS COMPARISON REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overall statistics
        f.write("## AGGREGATE STATISTICS\n\n")
        stats = compute_aggregate_statistics(model_data)
        f.write(stats.to_string(index=False))
        f.write("\n\n")

        # Per-profession comparison
        f.write("## PER-PROFESSION BIAS (Activation-based)\n\n")
        for model_name, df in model_data.items():
            f.write(f"\n### {model_name}\n")
            prof_avg = df.groupby('profession')['continuous_bias'].mean().sort_values(ascending=False)
            for prof, bias in prof_avg.items():
                direction = "MASCULINE" if bias > 0 else "FEMININE"
                f.write(f"  {prof:<25} {bias:+.4f} ({direction})\n")

        # Text embedding bias
        f.write("\n\n## TEXT EMBEDDING BIAS\n\n")
        for model_name, df in text_bias_data.items():
            f.write(f"\n### {model_name}\n")
            sorted_df = df.sort_values('text_embedding_bias', ascending=False)
            for _, row in sorted_df.iterrows():
                direction = "MASCULINE" if row['text_embedding_bias'] > 0 else "FEMININE"
                f.write(f"  {row['profession']:<25} {row['text_embedding_bias']:+.4f} ({direction})\n")

        f.write("\n" + "=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Compare bias across multiple models")

    parser.add_argument(
        "--results-dir",
        type=str,
        default="multi_model_results",
        help="Directory containing model results"
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=["sdxl", "sd15"],
        help="Model keys to compare"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for comparison plots"
    )

    parser.add_argument(
        "--professions",
        type=str,
        nargs="+",
        help="Specific professions to compare (default: all)"
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"CROSS-MODEL COMPARISON")
    print(f"{'='*70}\n")

    # Load data
    print("Loading results...")
    model_data = {}
    text_bias_data = {}

    for model_key in args.models:
        model_dir = results_dir / model_key

        bias_file = model_dir / "bias_analysis.csv"
        text_bias_file = model_dir / "text_bias_analysis.csv"

        if not bias_file.exists():
            print(f"Warning: No results found for {model_key} at {bias_file}")
            continue

        model_data[model_key] = load_bias_results(str(bias_file))

        if text_bias_file.exists():
            text_bias_data[model_key] = load_text_bias_results(str(text_bias_file))

    if not model_data:
        print("Error: No model data loaded!")
        return

    print(f"Loaded {len(model_data)} models\n")

    # Generate comparison plots
    print("Generating comparison plots...")

    # Model summary
    print("  - Model summary statistics...")
    plot_model_summary(model_data, output_dir / "model_summary.png")

    # Text bias comparison
    if text_bias_data:
        print("  - Text embedding bias comparison...")
        plot_text_bias_comparison(text_bias_data, output_dir / "text_bias_comparison.png")

    # Per-profession comparisons
    print("  - Per-profession comparisons...")
    professions_to_compare = args.professions
    if not professions_to_compare:
        # Get all professions from first model
        first_df = next(iter(model_data.values()))
        professions_to_compare = first_df['profession'].unique()

    profession_dir = output_dir / "by_profession"
    profession_dir.mkdir(exist_ok=True)

    for profession in professions_to_compare:
        print(f"    - {profession}")
        plot_profession_comparison(
            model_data,
            profession,
            profession_dir / f"{profession.lower().replace(' ', '_')}.png"
        )

    # Generate text report
    print("  - Summary report...")
    create_summary_report(
        model_data,
        text_bias_data,
        output_dir / "summary_report.txt"
    )

    print(f"\n{'='*70}")
    print(f"COMPARISON COMPLETE")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
