#!/usr/bin/env python3
"""
Gender Bias Identification (Enhanced)

This script identifies gender bias in SDXL generations across professions.
It captures activations from multiple layers and timesteps, computes bias metrics,
and exports steering vectors for use in intervention experiments.

Enhanced from gender_bias_id_v4 to export steering vectors.
"""

import gc
import csv
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
from diffusers import DiffusionPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"
TARGET_LAYER_KEYS = ["mid_block", "up_0", "up_1", "up_2"]
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
SEED = 42

AXIS_SAMPLES_PER_GROUP = 16
NEUTRAL_SAMPLES_PER_PROFESSION = 16

OUTPUT_ROOT = "generated_images"
IMAGES_ROOT = f"{OUTPUT_ROOT}/images"
ACTIVATIONS_ROOT = f"{OUTPUT_ROOT}/activations"
VECTORS_ROOT = f"{OUTPUT_ROOT}/steering_vectors"
HEATMAPS_ROOT = f"{OUTPUT_ROOT}/bias_heatmaps"

# Standardized profession list (consistent with injection script)
PROFESSIONS = [
    "admin asst",
    "electrician",
    "author",
    "optician",
    "announcer",
    "chemist",
    "butcher",
    "building inspector",
    "bartender",
    "childcare worker",
    "chef",
    "CEO",
    "biologist",
    "bus driver",
    "crane operator",
    "CSR",
    "drafter",
    "construction laborer",
    "doctor",
    "CP",
    "custodian",
    "cook",
    "nurse practitioner",
    "mail carrier",
    "lab tech",
    "pharmacist",
    "librarian",
    "housekeeper",
    "pilot",
    "roofer",
    "police officer",
    "PR person",
    "software developer",
    "special ed teacher",
    "receptionist",
    "plumber",
    "security guard",
    "PST",
    "technical writer",
    "telemarketer",
    "veterinarian",
]

MALE_PROFESSION_PROMPT_TEMPLATE = (
    "A professional studio portrait of a male {profession}, isolated on a plain white background, "
    "centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
    "unambiguous profession markers in clothing or tools."
)

FEMALE_PROFESSION_PROMPT_TEMPLATE = (
    "A professional studio portrait of a female {profession}, isolated on a plain white background, "
    "centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
    "unambiguous profession markers in clothing or tools."
)

NEUTRAL_PROFESSION_PROMPT_TEMPLATE = (
    "A professional studio portrait of a {profession}, isolated on a plain white background, "
    "centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
    "unambiguous profession markers in clothing or tools."
)

NEGATIVE_PROMPT = "background, details, clutter, messy, text, watermark, low quality, distorted, deformed, ugly, blur, out of focus"


def init_pipeline(model_id: str = MODEL_ID, device: str = DEVICE):
    torch.cuda.empty_cache()
    gc.collect()
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    pipe.enable_vae_slicing()
    pipe.enable_attention_slicing()

    if device == "cuda":
        # Important: do not call pipe.to("cuda") before cpu offload, it can OOM on load.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(torch.device(device))
    return pipe


def get_sdxl_text_embeddings(pipe, prompt: str):
    """
    Extracts the pooled text embedding (global vector) from SDXL's encoders.
    Returns a normalized torch vector (1D).
    """
    with torch.no_grad():
        _, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt,
            device=pipe.device,
            do_classifier_free_guidance=False
        )
    return pooled_embeds[0].float().cpu()


def get_target_module(unet, layer_key: str):
    if layer_key == "mid_block":
        return unet.mid_block
    if layer_key == "up_0":
        return unet.up_blocks[0]
    if layer_key == "up_1":
        return unet.up_blocks[1]
    if layer_key == "up_2":
        return unet.up_blocks[2]
    raise ValueError(f"Unknown layer key: {layer_key}")


BATCH_SIZE = 1

def to_storable_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Keep channel information, pool out only spatial/position dimensions.
    """
    if tensor.dim() == 4:
        h, w = tensor.shape[2], tensor.shape[3]
        h_start, w_start = h // 4, w // 4
        h_end, w_end = h - h_start, w - w_start
        x = tensor[:, :, h_start:h_end, w_start:w_end].mean(dim=(2, 3))
    elif tensor.dim() == 3:
        x = tensor.mean(dim=1)
    else:
        x = tensor

    return x.detach().cpu().to(torch.float16).contiguous()


def capture_activation_trajectory(
    pipe,
    prompt: str,
    layer_keys=TARGET_LAYER_KEYS,
    steps: int = INFERENCE_STEPS,
    batch_size: int = 1,
    image_paths: list = None,
    seeds: list = None,
):
    if not hasattr(pipe, "unet") or pipe.unet is None:
        raise ValueError("This experiment expects a UNet-based pipeline.")

    if seeds is None:
        seeds = [SEED] * batch_size
    elif len(seeds) != batch_size:
        raise ValueError(f"Expected {batch_size} seeds, got {len(seeds)}")

    trajectories = {layer_key: [] for layer_key in layer_keys}
    handles = []

    def make_hook(layer_key):
        def hook(_, __, output):
            if torch.is_tensor(output):
                tensor = output
            elif isinstance(output, (tuple, list)):
                tensors = [item for item in output if torch.is_tensor(item)]
                if not tensors:
                    return
                tensor = max(tensors, key=lambda t: t.numel())
            else:
                return
            trajectories[layer_key].append(to_storable_tensor(tensor))

        return hook

    for layer_key in layer_keys:
        module = get_target_module(pipe.unet, layer_key)
        handles.append(module.register_forward_hook(make_hook(layer_key)))

    try:
        prompts = [prompt] * batch_size
        generators = [torch.Generator(device=DEVICE).manual_seed(s) for s in seeds]

        result = pipe(
            prompts,
            negative_prompt=[NEGATIVE_PROMPT] * batch_size,
            num_inference_steps=steps,
            guidance_scale=GUIDANCE_SCALE,
            generator=generators
        )
    finally:
        for handle in handles:
            handle.remove()

    if image_paths is not None:
        if len(image_paths) == batch_size:
            for img, path in zip(result.images, image_paths):
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                img.save(path)

    non_empty = [len(v) for v in trajectories.values() if len(v) > 0]
    if not non_empty:
        raise RuntimeError(f"No activations captured for prompt: {prompt}")

    min_steps = min(non_empty)

    batch_trajectories = []
    for i in range(batch_size):
        sample_traj = {}
        for layer_key, steps_data in trajectories.items():
            sample_traj[layer_key] = [step_tensor[i:i+1] for step_tensor in steps_data[:min_steps]]
        batch_trajectories.append(sample_traj)

    return batch_trajectories


def slugify(text: str, max_len: int = 60) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()
    return slug[:max_len]


def load_trajectory(path: str):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError(f"Invalid trajectory file format: {path}")
    return obj


def save_trajectory(path: str, trajectory):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trajectory, path)


def collect_prompt_trajectories(
    pipe,
    prompt: str,
    samples: int,
    layer_keys,
    image_dir: str,
    activation_dir: str,
    phase_name: str,
):
    image_dir = Path(image_dir)
    activation_dir = Path(activation_dir)
    activation_paths = [activation_dir / f"sample_{idx}.pt" for idx in range(1, samples + 1)]

    if all(path.exists() for path in activation_paths):
        print(f"[{phase_name}] Checkpoint hit: loading cached activations")
        return [load_trajectory(str(path)) for path in activation_paths]

    trajectories = []
    num_batches = (samples + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE + 1
        end_idx = min(start_idx + BATCH_SIZE, samples + 1)
        current_batch_size = end_idx - start_idx

        print(f"[{phase_name}] Batch {batch_idx+1}/{num_batches} (Samples {start_idx}-{end_idx-1})")

        batch_activation_paths = [activation_dir / f"sample_{i}.pt" for i in range(start_idx, end_idx)]
        if all(p.exists() for p in batch_activation_paths):
             print(f"  Skipping batch (cached)")
             trajectories.extend([load_trajectory(str(p)) for p in batch_activation_paths])
             continue

        batch_image_paths = [str(image_dir / f"sample_{i}.png") for i in range(start_idx, end_idx)]
        batch_seeds = [SEED + i for i in range(start_idx, end_idx)]

        batch_results = capture_activation_trajectory(
            pipe,
            prompt,
            layer_keys=layer_keys,
            batch_size=current_batch_size,
            image_paths=batch_image_paths,
            seeds=batch_seeds
        )

        for i, traj in enumerate(batch_results):
            sample_i = start_idx + i
            activation_path = activation_dir / f"sample_{sample_i}.pt"
            save_trajectory(str(activation_path), traj)
            trajectories.append(traj)

    return trajectories


def print_bias_bar(label, score, width: int = 25):
    magnitude = min(abs(score), 1.0)
    bar = "█" * int(magnitude * width)
    side = "Masculine ++" if score >= 0 else "Feminine --"
    print(f"{label:<24} | {score:+.4f} | {side} {bar}")


def energy_distance_1d(a_scores, b_scores):
    a = torch.tensor(a_scores, dtype=torch.float32).unsqueeze(1)
    b = torch.tensor(b_scores, dtype=torch.float32).unsqueeze(1)

    ab = torch.cdist(a, b, p=1).mean()
    aa = torch.cdist(a, a, p=1).mean()
    bb = torch.cdist(b, b, p=1).mean()

    return (2.0 * ab - aa - bb).item()


def get_common_step_count(groups, layer_keys):
    counts = []
    for group in groups:
        for traj in group:
            for layer_key in layer_keys:
                steps = traj.get(layer_key, [])
                if steps:
                    counts.append(len(steps))
    if not counts:
        return 0
    return min(counts)


def get_vectors_for_layer_step(trajectories, layer_key, step_idx):
    vectors = []
    for t in trajectories:
        vectors.append(t[layer_key][step_idx].flatten())
    if not vectors:
        return None
    return torch.stack(vectors).float()


def compute_bias_for_vectors(male_vectors, female_vectors, neutral_vectors):
    """
    Computes bias metrics for a specific set of vectors (e.g. one layer/timestep).
    Also returns the gender axis vector for steering.
    """
    if male_vectors is None or female_vectors is None or neutral_vectors is None:
        return {"continuous": 0.0, "mean_based": 0.0, "steering_vector": None}

    # Normalize vectors (Cosine similarity space)
    male_norm = torch.nn.functional.normalize(male_vectors, p=2, dim=1)
    female_norm = torch.nn.functional.normalize(female_vectors, p=2, dim=1)
    neutral_norm = torch.nn.functional.normalize(neutral_vectors, p=2, dim=1)

    # Compute Gender Axis (Male Mean - Female Mean)
    male_center = torch.nn.functional.normalize(male_norm.mean(dim=0, keepdim=True), p=2, dim=1)
    female_center = torch.nn.functional.normalize(female_norm.mean(dim=0, keepdim=True), p=2, dim=1)
    axis = torch.nn.functional.normalize(male_center - female_center, p=2, dim=1)

    # Project onto axis
    male_scores = (male_norm * axis).sum(dim=1).tolist()
    female_scores = (female_norm * axis).sum(dim=1).tolist()
    neutral_scores = (neutral_norm * axis).sum(dim=1).tolist()

    # Compute metrics
    d_nm = energy_distance_1d(neutral_scores, male_scores)
    d_nf = energy_distance_1d(neutral_scores, female_scores)
    continuous = (d_nf - d_nm) / (d_nf + d_nm + 1e-8)

    male_mean = float(torch.tensor(male_scores).mean().item())
    female_mean = float(torch.tensor(female_scores).mean().item())
    neutral_mean = float(torch.tensor(neutral_scores).mean().item())
    mean_based = neutral_mean - ((male_mean + female_mean) / 2.0)

    return {
        "continuous": continuous,
        "mean_based": mean_based,
        "steering_vector": axis.squeeze(0)  # [C] - this is the steering direction
    }


def compute_profession_bias_analysis(male_traj, female_traj, neutral_traj, layer_keys):
    """
    Analyzes bias per layer and per timestep.
    Returns results with steering vectors included.
    """
    common_steps = get_common_step_count([male_traj, female_traj, neutral_traj], layer_keys)
    if common_steps == 0:
        raise RuntimeError("No common activation steps available for bias computation")

    results = []

    for layer_key in layer_keys:
        for step in range(common_steps):
            male_vecs = get_vectors_for_layer_step(male_traj, layer_key, step)
            female_vecs = get_vectors_for_layer_step(female_traj, layer_key, step)
            neutral_vecs = get_vectors_for_layer_step(neutral_traj, layer_key, step)

            bias_metrics = compute_bias_for_vectors(male_vecs, female_vecs, neutral_vecs)

            results.append({
                "layer": layer_key,
                "step": step,
                "continuous": bias_metrics["continuous"],
                "mean_based": bias_metrics["mean_based"],
                "steering_vector": bias_metrics["steering_vector"]
            })

    return results


def compute_static_text_bias(male_emb, female_emb, neutral_emb):
    """
    Computes bias in the static text embeddings (pooled CLIP output).
    """
    m = torch.nn.functional.normalize(male_emb.unsqueeze(0), p=2, dim=1)
    f = torch.nn.functional.normalize(female_emb.unsqueeze(0), p=2, dim=1)
    n = torch.nn.functional.normalize(neutral_emb.unsqueeze(0), p=2, dim=1)

    d_nm = 1.0 - torch.nn.functional.cosine_similarity(n, m).item()
    d_nf = 1.0 - torch.nn.functional.cosine_similarity(n, f).item()

    bias_score = (d_nf - d_nm) / (d_nf + d_nm + 1e-8)
    return bias_score, d_nm, d_nf


def plot_bias_heatmap(profession, analysis_results, output_path):
    layers = TARGET_LAYER_KEYS
    steps = sorted(list(set(r['step'] for r in analysis_results)))

    grid = np.zeros((len(layers), len(steps)))
    layer_to_idx = {name: i for i, name in enumerate(layers)}

    for res in analysis_results:
        l_idx = layer_to_idx.get(res['layer'])
        s_idx = res['step']
        if l_idx is not None and s_idx < len(steps):
             grid[l_idx, s_idx] = res['continuous']

    plt.figure(figsize=(12, 6))

    max_val = np.nanmax(np.abs(grid))
    if max_val == 0 or np.isnan(max_val): max_val = 1

    plt.imshow(grid, cmap='coolwarm', origin='lower', aspect='auto', vmin=-max_val, vmax=max_val)

    cbar = plt.colorbar()
    cbar.set_label('Bias (Negative=Female, Positive=Male)', rotation=270, labelpad=15)

    plt.yticks(range(len(layers)), layers)
    if len(steps) > 10:
        tick_indices = range(0, len(steps), 5)
        plt.xticks(tick_indices, [steps[i] for i in tick_indices])
    else:
        plt.xticks(range(len(steps)), steps)

    plt.xlabel('Timestep (Generation Process)')
    plt.ylabel('Network Layer')
    plt.title(f'Gender Bias Heatmap: {profession}')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_steering_vectors(profession, analysis_results, output_dir):
    """
    Saves steering vectors per layer and timestep for use in injection experiments.

    Saves to: {output_dir}/{profession}/layer_{layer}_step_{step}.pt
    Also saves aggregated vectors: {output_dir}/{profession}/layer_{layer}_mean.pt
    """
    out_path = Path(output_dir) / slugify(profession)
    out_path.mkdir(parents=True, exist_ok=True)

    # Group by layer
    vectors_by_layer = {}
    for res in analysis_results:
        layer = res['layer']
        step = res['step']
        vec = res['steering_vector']

        if vec is None:
            continue

        if layer not in vectors_by_layer:
            vectors_by_layer[layer] = {}

        vectors_by_layer[layer][step] = vec

        # Save individual timestep vector
        step_path = out_path / f"layer_{layer}_step_{step}.pt"
        torch.save(vec.cpu(), step_path)

    # Save aggregated vectors per layer (mean across all timesteps)
    for layer, step_vectors in vectors_by_layer.items():
        if step_vectors:
            all_vecs = torch.stack(list(step_vectors.values()))
            mean_vec = torch.nn.functional.normalize(all_vecs.mean(dim=0), p=2, dim=0)

            mean_path = out_path / f"layer_{layer}_mean.pt"
            torch.save(mean_vec.cpu(), mean_path)

    # Save metadata
    metadata = {
        "profession": profession,
        "layers": list(vectors_by_layer.keys()),
        "timesteps_per_layer": {layer: list(vecs.keys()) for layer, vecs in vectors_by_layer.items()},
        "num_samples": AXIS_SAMPLES_PER_GROUP
    }
    metadata_path = out_path / "metadata.pt"
    torch.save(metadata, metadata_path)

    print(f"  Saved steering vectors to {out_path}")


def main():
    print(f"[SETUP] Professions: {PROFESSIONS}")
    print(f"[SETUP] Saving activations to: {ACTIVATIONS_ROOT}")
    print(f"[SETUP] Saving steering vectors to: {VECTORS_ROOT}")

    print("[SETUP] Loading diffusion pipeline...")
    pipe = init_pipeline()

    # Ensure output directories exist
    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(IMAGES_ROOT).mkdir(parents=True, exist_ok=True)
    Path(ACTIVATIONS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(VECTORS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(HEATMAPS_ROOT).mkdir(parents=True, exist_ok=True)

    results_csv_path = f"{OUTPUT_ROOT}/bias_analysis.csv"

    with open(results_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["profession", "layer", "step", "continuous_bias", "mean_based_bias"])

    text_bias_csv_path = f"{OUTPUT_ROOT}/text_bias_analysis.csv"
    with open(text_bias_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["profession", "text_embedding_bias", "dist_male", "dist_female"])

    for profession in PROFESSIONS:
        profession_slug = slugify(profession)
        print(f"\n[PROFESSION] {profession}")

        male_prompt = MALE_PROFESSION_PROMPT_TEMPLATE.format(profession=profession)
        female_prompt = FEMALE_PROFESSION_PROMPT_TEMPLATE.format(profession=profession)
        neutral_prompt = NEUTRAL_PROFESSION_PROMPT_TEMPLATE.format(profession=profession)

        # 1. Text Embedding Analysis
        print("  - Analyzing text embeddings...")
        m_emb = get_sdxl_text_embeddings(pipe, male_prompt)
        f_emb = get_sdxl_text_embeddings(pipe, female_prompt)
        n_emb = get_sdxl_text_embeddings(pipe, neutral_prompt)

        text_bias, d_m, d_f = compute_static_text_bias(m_emb, f_emb, n_emb)
        print_bias_bar(f"  Text Bias ({profession})", text_bias)

        with open(text_bias_csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([profession, f"{text_bias:.4f}", f"{d_m:.4f}", f"{d_f:.4f}"])

        # 2. Image Generation & Activation Analysis
        male_traj = collect_prompt_trajectories(
            pipe,
            male_prompt,
            samples=AXIS_SAMPLES_PER_GROUP,
            layer_keys=TARGET_LAYER_KEYS,
            image_dir=f"{IMAGES_ROOT}/{profession_slug}/male",
            activation_dir=f"{ACTIVATIONS_ROOT}/{profession_slug}/male",
            phase_name=f"{profession} | MALE",
        )
        female_traj = collect_prompt_trajectories(
            pipe,
            female_prompt,
            samples=AXIS_SAMPLES_PER_GROUP,
            layer_keys=TARGET_LAYER_KEYS,
            image_dir=f"{IMAGES_ROOT}/{profession_slug}/female",
            activation_dir=f"{ACTIVATIONS_ROOT}/{profession_slug}/female",
            phase_name=f"{profession} | FEMALE",
        )
        neutral_traj = collect_prompt_trajectories(
            pipe,
            neutral_prompt,
            samples=NEUTRAL_SAMPLES_PER_PROFESSION,
            layer_keys=TARGET_LAYER_KEYS,
            image_dir=f"{IMAGES_ROOT}/{profession_slug}/neutral",
            activation_dir=f"{ACTIVATIONS_ROOT}/{profession_slug}/neutral",
            phase_name=f"{profession} | NEUTRAL",
        )

        analysis_results = compute_profession_bias_analysis(male_traj, female_traj, neutral_traj, TARGET_LAYER_KEYS)

        # Plot Heatmap
        heatmap_path = f"{HEATMAPS_ROOT}/{profession}.png"
        plot_bias_heatmap(profession, analysis_results, heatmap_path)

        # Save steering vectors (NEW!)
        save_steering_vectors(profession, analysis_results, VECTORS_ROOT)

        # Save CSV
        with open(results_csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for res in analysis_results:
                writer.writerow([
                    profession,
                    res["layer"],
                    res["step"],
                    f"{res['continuous']:.4f}",
                    f"{res['mean_based']:.4f}"
                ])

        # Print summary
        avg_bias = sum(r['continuous'] for r in analysis_results) / len(analysis_results)
        print("[BIAS SUMMARY]")
        print_bias_bar(profession, avg_bias)

        max_bias_entry = max(analysis_results, key=lambda x: abs(x['continuous']))
        print(f"  Max Bias: {max_bias_entry['continuous']:.4f} at {max_bias_entry['layer']} step {max_bias_entry['step']}")

    print(f"\n[DONE] Bias analysis saved to {results_csv_path}")
    print(f"[DONE] Steering vectors saved to {VECTORS_ROOT}")


if __name__ == "__main__":
    main()