import os
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_activation_heatmap(activation, title, output_path):
    """
    Plots a heatmap for a 2D activation matrix (timesteps x features) and saves it.
    """
    plt.figure(figsize=(10, 4))
    plt.imshow(activation.T, aspect='auto', cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Activation')
    plt.xlabel('Timestep')
    plt.ylabel('Feature')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    input_root = Path('outputs/debiased_robust_report')
    output_root = Path('outputs/debiased_robust_report/steering_heatmaps')
    output_root.mkdir(parents=True, exist_ok=True)

    poles = [
        ('male_steps.pt', 'Masculine'),
        ('female_steps.pt', 'Feminine'),
        ('neutral_steps.pt', 'Neutral'),
    ]

    for profession_dir in input_root.iterdir():
        if not profession_dir.is_dir():
            continue
        prof_name = profession_dir.name
        for fname, pole in poles:
            act_path = profession_dir / fname
            if act_path.exists():
                activations = torch.load(act_path, map_location='cpu')  # shape: [layers, features] or [timesteps, features]
                # If activations are 2D (timesteps x features), treat as one layer
                if activations.dim() == 2:
                    plot_activation_heatmap(
                        activations,
                        f'{prof_name} - {pole}',
                        output_root / f'{prof_name}_{pole.lower()}_all_layers.png'
                    )
                # If activations are 3D (layers x timesteps x features)
                elif activations.dim() == 3:
                    for layer_idx, layer_act in enumerate(activations):
                        plot_activation_heatmap(
                            layer_act,
                            f'{prof_name} - {pole} - Layer {layer_idx}',
                            output_root / f'{prof_name}_{pole.lower()}_layer{layer_idx}.png'
                        )
                # If activations are [timesteps, features] per layer (as in [layers, features])
                elif activations.dim() == 2 and activations.shape[0] < activations.shape[1]:
                    # Could be [layers, features], treat each as a feature vector
                    for layer_idx, layer_vec in enumerate(activations):
                        plot_activation_heatmap(
                            layer_vec[None, :],
                            f'{prof_name} - {pole} - Layer {layer_idx}',
                            output_root / f'{prof_name}_{pole.lower()}_layer{layer_idx}.png'
                        )
                else:
                    print(f'Unknown activation shape for {act_path}: {activations.shape}')

if __name__ == '__main__':
    main()
