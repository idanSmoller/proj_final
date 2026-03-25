#!/usr/bin/env python3
"""
Visualize CLIP Text Embedding Bias vs Activation Bias

Creates a grouped bar chart comparing text embedding bias and activation-based bias.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def visualize_bias_comparison(model_dir):
    """Create comparison visualization of text and activation bias."""

    # Load data
    text_bias_df = pd.read_csv(Path(model_dir) / 'text_bias_analysis.csv')
    activation_bias_df = pd.read_csv(Path(model_dir) / 'bias_analysis.csv')

    # Calculate average activation bias per profession
    activation_avg = activation_bias_df.groupby('profession')['continuous_bias'].mean()

    # Merge datasets
    merged = text_bias_df.merge(
        activation_avg.reset_index().rename(columns={'continuous_bias': 'activation_bias'}),
        on='profession',
        how='left'
    )

    # Sort by text embedding bias
    merged = merged.sort_values('text_embedding_bias', ascending=True)

    professions = merged['profession'].values
    text_bias = merged['text_embedding_bias'].values
    activation_bias = merged['activation_bias'].values

    # Create figure
    fig, ax = plt.subplots(figsize=(12, max(8, len(professions) * 0.3)))

    # Bar positions
    y_pos = np.arange(len(professions))
    bar_height = 0.35

    # Create bars
    bars1 = ax.barh(y_pos - bar_height/2, text_bias, bar_height,
                    label='Text Embedding Bias', alpha=0.8, edgecolor='black', linewidth=0.5)
    bars2 = ax.barh(y_pos + bar_height/2, activation_bias, bar_height,
                    label='Activation Bias', alpha=0.8, edgecolor='black', linewidth=0.5)

    # Color bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        bar1.set_color('#1f77b4' if text_bias[i] < 0 else '#d62728')
        bar2.set_color('#9467bd' if activation_bias[i] < 0 else '#ff7f0e')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(professions)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Bias Score (negative=feminine, positive=masculine)', fontweight='bold', fontsize=12)
    ax.set_title('Text Embedding Bias vs Activation Bias by Profession', fontweight='bold', fontsize=14)

    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Text Embedding (Feminine)'),
        Patch(facecolor='#d62728', edgecolor='black', label='Text Embedding (Masculine)'),
        Patch(facecolor='#9467bd', edgecolor='black', label='Activation (Feminine)'),
        Patch(facecolor='#ff7f0e', edgecolor='black', label='Activation (Masculine)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = Path(__file__).parent / 'text_bias_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print("✓ Visualization saved to: {}".format(output_path))

    plt.show()


if __name__ == '__main__':
    model_dir = '/home/idan.smoller/proj_final/multi_model_results/sdxl'
    visualize_bias_comparison(model_dir)
