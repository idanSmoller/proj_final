#!/usr/bin/env python3
"""
Gemini-guided Bias Identification for SDXL

This script identifies bias in SDXL generations across professions using
Gemini-proposed semantic axes instead of hardcoded male/female prompts.

For each profession:
1. Gemini proposes 3 plausible bias axes.
2. Each axis contains:
   - prompt_a
   - prompt_b
   - base_prompt
3. The script generates images and captures activations for each of the three prompts.
4. It computes bias metrics per layer and timestep.
5. It exports heatmaps, steering vectors, CSV summaries, and Gemini outputs.
"""

import gc
import csv
import json
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np
import requests
from diffusers import DiffusionPipeline


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"
TARGET_LAYER_KEYS = ["mid_block", "up_0", "up_1", "up_2"]
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
SEED = 42

AXIS_SAMPLES_PER_GROUP = 16
BASE_SAMPLES_PER_AXIS = 16

OUTPUT_ROOT = "generated_images"
IMAGES_ROOT = f"{OUTPUT_ROOT}/images"
ACTIVATIONS_ROOT = f"{OUTPUT_ROOT}/activations"
VECTORS_ROOT = f"{OUTPUT_ROOT}/steering_vectors"
HEATMAPS_ROOT = f"{OUTPUT_ROOT}/bias_heatmaps"
AXES_ROOT = f"{OUTPUT_ROOT}/gemini_axes"

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

PROFESSIONS = [
    # "Nurse",
    # "Fire Fighter",
    # "Programmer",
    # "Receptionist",
    # "Kindergarten Teacher",
    # "Doctor",
    # "Lawyer",
    # "Engineer",
    # "Scientist",
    # "Accountant",
    # "Chef",
    # "Architect",
    # "CEO",
    # "Pilot",
    # "Flight Attendant",
    "Shoemaker",
]

NEGATIVE_PROMPT = (
    "background, details, clutter, messy, text, watermark, low quality, "
    "distorted, deformed, ugly, blur, out of focus"
)

BATCH_SIZE = 1


def init_pipeline(model_id: str = MODEL_ID, device: str = DEVICE):
    torch.cuda.empty_cache()
    gc.collect()
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )

    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()
    if hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe


def slugify(text: str, max_len: int = 80) -> str:
    slug = "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()
    return slug[:max_len]


def get_sdxl_text_embeddings(pipe, prompt: str):
    with torch.no_grad():
        execution_device = getattr(pipe, "_execution_device", None)
        if execution_device is None:
            execution_device = torch.device(DEVICE)

        _, _, pooled_embeds, _ = pipe.encode_prompt(
            prompt,
            device=execution_device,
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


def to_storable_tensor(tensor: torch.Tensor) -> torch.Tensor:
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
            generator=generators,
        )
    finally:
        for handle in handles:
            handle.remove()

    if image_paths is not None and len(image_paths) == batch_size:
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

        print(f"[{phase_name}] Batch {batch_idx + 1}/{num_batches} (Samples {start_idx}-{end_idx - 1})")

        batch_activation_paths = [activation_dir / f"sample_{i}.pt" for i in range(start_idx, end_idx)]
        if all(p.exists() for p in batch_activation_paths):
            print("  Skipping batch (cached)")
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
            seeds=batch_seeds,
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
    side = "Pole A ++" if score >= 0 else "Pole B --"
    print(f"{label:<32} | {score:+.4f} | {side} {bar}")


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


def compute_bias_for_vectors(pole_a_vectors, pole_b_vectors, base_vectors):
    if pole_a_vectors is None or pole_b_vectors is None or base_vectors is None:
        return {"continuous": 0.0, "mean_based": 0.0, "steering_vector": None}

    pole_a_norm = torch.nn.functional.normalize(pole_a_vectors, p=2, dim=1)
    pole_b_norm = torch.nn.functional.normalize(pole_b_vectors, p=2, dim=1)
    base_norm = torch.nn.functional.normalize(base_vectors, p=2, dim=1)

    pole_a_center = torch.nn.functional.normalize(pole_a_norm.mean(dim=0, keepdim=True), p=2, dim=1)
    pole_b_center = torch.nn.functional.normalize(pole_b_norm.mean(dim=0, keepdim=True), p=2, dim=1)
    axis = torch.nn.functional.normalize(pole_a_center - pole_b_center, p=2, dim=1)

    pole_a_scores = (pole_a_norm * axis).sum(dim=1).tolist()
    pole_b_scores = (pole_b_norm * axis).sum(dim=1).tolist()
    base_scores = (base_norm * axis).sum(dim=1).tolist()

    d_base_a = energy_distance_1d(base_scores, pole_a_scores)
    d_base_b = energy_distance_1d(base_scores, pole_b_scores)
    continuous = (d_base_b - d_base_a) / (d_base_b + d_base_a + 1e-8)

    pole_a_mean = float(torch.tensor(pole_a_scores).mean().item())
    pole_b_mean = float(torch.tensor(pole_b_scores).mean().item())
    base_mean = float(torch.tensor(base_scores).mean().item())
    mean_based = base_mean - ((pole_a_mean + pole_b_mean) / 2.0)

    return {
        "continuous": continuous,
        "mean_based": mean_based,
        "steering_vector": axis.squeeze(0),
    }


def compute_axis_bias_analysis(pole_a_traj, pole_b_traj, base_traj, layer_keys):
    common_steps = get_common_step_count([pole_a_traj, pole_b_traj, base_traj], layer_keys)
    if common_steps == 0:
        raise RuntimeError("No common activation steps available for bias computation")

    results = []

    for layer_key in layer_keys:
        for step in range(common_steps):
            pole_a_vecs = get_vectors_for_layer_step(pole_a_traj, layer_key, step)
            pole_b_vecs = get_vectors_for_layer_step(pole_b_traj, layer_key, step)
            base_vecs = get_vectors_for_layer_step(base_traj, layer_key, step)

            bias_metrics = compute_bias_for_vectors(pole_a_vecs, pole_b_vecs, base_vecs)

            results.append({
                "layer": layer_key,
                "step": step,
                "continuous": bias_metrics["continuous"],
                "mean_based": bias_metrics["mean_based"],
                "steering_vector": bias_metrics["steering_vector"],
            })

    return results


def compute_static_text_bias(pole_a_emb, pole_b_emb, base_emb):
    a = torch.nn.functional.normalize(pole_a_emb.unsqueeze(0), p=2, dim=1)
    b = torch.nn.functional.normalize(pole_b_emb.unsqueeze(0), p=2, dim=1)
    n = torch.nn.functional.normalize(base_emb.unsqueeze(0), p=2, dim=1)

    d_base_a = 1.0 - torch.nn.functional.cosine_similarity(n, a).item()
    d_base_b = 1.0 - torch.nn.functional.cosine_similarity(n, b).item()

    bias_score = (d_base_b - d_base_a) / (d_base_b + d_base_a + 1e-8)
    return bias_score, d_base_a, d_base_b


def plot_bias_heatmap(profession, axis_name, analysis_results, output_path):
    layers = TARGET_LAYER_KEYS
    steps = sorted(list(set(r["step"] for r in analysis_results)))

    grid = np.zeros((len(layers), len(steps)))
    layer_to_idx = {name: i for i, name in enumerate(layers)}

    for res in analysis_results:
        l_idx = layer_to_idx.get(res["layer"])
        s_idx = res["step"]
        if l_idx is not None and s_idx < len(steps):
            grid[l_idx, s_idx] = res["continuous"]

    plt.figure(figsize=(12, 6))

    max_val = np.nanmax(np.abs(grid))
    if max_val == 0 or np.isnan(max_val):
        max_val = 1

    plt.imshow(grid, cmap="coolwarm", origin="lower", aspect="auto", vmin=-max_val, vmax=max_val)

    cbar = plt.colorbar()
    cbar.set_label("Bias polarity", rotation=270, labelpad=15)

    plt.yticks(range(len(layers)), layers)
    if len(steps) > 10:
        tick_indices = range(0, len(steps), 5)
        plt.xticks(tick_indices, [steps[i] for i in tick_indices])
    else:
        plt.xticks(range(len(steps)), steps)

    plt.xlabel("Timestep")
    plt.ylabel("Network Layer")
    plt.title(f"Bias Heatmap: {profession} | Axis: {axis_name}")

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_steering_vectors(profession, axis_name, analysis_results, output_dir):
    out_path = Path(output_dir) / slugify(profession) / slugify(axis_name)
    out_path.mkdir(parents=True, exist_ok=True)

    vectors_by_layer = {}
    for res in analysis_results:
        layer = res["layer"]
        step = res["step"]
        vec = res["steering_vector"]

        if vec is None:
            continue

        if layer not in vectors_by_layer:
            vectors_by_layer[layer] = {}

        vectors_by_layer[layer][step] = vec

        step_path = out_path / f"layer_{layer}_step_{step}.pt"
        torch.save(vec.cpu(), step_path)

    for layer, step_vectors in vectors_by_layer.items():
        if step_vectors:
            all_vecs = torch.stack(list(step_vectors.values()))
            mean_vec = torch.nn.functional.normalize(all_vecs.mean(dim=0), p=2, dim=0)

            mean_path = out_path / f"layer_{layer}_mean.pt"
            torch.save(mean_vec.cpu(), mean_path)

    metadata = {
        "profession": profession,
        "axis_name": axis_name,
        "layers": list(vectors_by_layer.keys()),
        "timesteps_per_layer": {layer: list(vecs.keys()) for layer, vecs in vectors_by_layer.items()},
        "num_samples": AXIS_SAMPLES_PER_GROUP,
    }
    metadata_path = out_path / "metadata.pt"
    torch.save(metadata, metadata_path)

    print(f"    Saved steering vectors to {out_path}")


def extract_json_block(text: str):
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Gemini output did not include a JSON object")

    return json.loads(cleaned[start:end + 1])


def build_gemini_axis_prompt(target_concept: str) -> str:
    return f"""
You are helping with fairness research for image generation prompts.

Target concept: {target_concept}

Identify exactly 3 plausible social or semantic bias axes that may affect how this concept is portrayed.
Examples:
- male vs female
- rich vs poor
- powerful vs powerless
- glamorous vs ordinary
- young vs old

For each axis, provide:
1. rank
2. axis_name
3. pole_a
4. pole_b
5. relevance
6. base_prompt
7. prompt_a
8. prompt_b

Return STRICT JSON only in this schema:
{{
  "concept": "{target_concept}",
  "axes": [
    {{
      "rank": 1,
      "axis_name": "",
      "pole_a": "",
      "pole_b": "",
      "relevance": "",
      "base_prompt": "",
      "prompt_a": "",
      "prompt_b": ""
    }},
    {{
      "rank": 2,
      "axis_name": "",
      "pole_a": "",
      "pole_b": "",
      "relevance": "",
      "base_prompt": "",
      "prompt_a": "",
      "prompt_b": ""
    }},
    {{
      "rank": 3,
      "axis_name": "",
      "pole_a": "",
      "pole_b": "",
      "relevance": "",
      "base_prompt": "",
      "prompt_a": "",
      "prompt_b": ""
    }}
  ]
}}

Rules:
- Keep prompts photorealistic.
- Preserve the target concept.
- Make the axes plausible and visually meaningful.
- Return JSON only.
""".strip()


def request_gemini_axes(target_concept: str, api_key: str):
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    prompt = build_gemini_axis_prompt(target_concept)
    url = GEMINI_API_URL.format(model=GEMINI_MODEL)

    response = requests.post(
        f"{url}?key={api_key}",
        json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.5,
                "topP": 0.9,
                "maxOutputTokens": 1600,
            },
        },
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

    print("\n" + "=" * 100)
    print(f"[GEMINI RAW OUTPUT] {target_concept}")
    print("=" * 100)
    print(raw_text)

    parsed = extract_json_block(raw_text)
    axes = parsed.get("axes", [])
    if not isinstance(axes, list) or len(axes) == 0:
        raise ValueError("Gemini returned no axes")

    print("\n[GEMINI PARSED AXES]")
    print(json.dumps(axes, indent=2, ensure_ascii=False))

    return prompt, axes, raw_text, parsed, data


def fallback_axes(concept: str):
    base_prompt = (
        f"A professional studio portrait of a {concept}, isolated on a plain white background, "
        f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
        f"unambiguous profession markers in clothing or tools."
    )
    return [
        {
            "rank": 1,
            "axis_name": "gender",
            "pole_a": "male",
            "pole_b": "female",
            "relevance": "Common visual stereotype axis.",
            "base_prompt": base_prompt,
            "prompt_a": (
                f"A professional studio portrait of a male {concept}, isolated on a plain white background, "
                f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
                f"unambiguous profession markers in clothing or tools."
            ),
            "prompt_b": (
                f"A professional studio portrait of a female {concept}, isolated on a plain white background, "
                f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
                f"unambiguous profession markers in clothing or tools."
            ),
        },
        {
            "rank": 2,
            "axis_name": "status",
            "pole_a": "high-status",
            "pole_b": "ordinary",
            "relevance": "Many professions are visually idealized through status cues.",
            "base_prompt": base_prompt,
            "prompt_a": (
                f"A professional studio portrait of a highly prestigious {concept}, isolated on a plain white background, "
                f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
                f"unambiguous profession markers in clothing or tools."
            ),
            "prompt_b": (
                f"A professional studio portrait of an ordinary working {concept}, isolated on a plain white background, "
                f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
                f"unambiguous profession markers in clothing or tools."
            ),
        },
        {
            "rank": 3,
            "axis_name": "age",
            "pole_a": "young",
            "pole_b": "older",
            "relevance": "Many professions are stereotyped by age presentation.",
            "base_prompt": base_prompt,
            "prompt_a": (
                f"A professional studio portrait of a young {concept}, isolated on a plain white background, "
                f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
                f"unambiguous profession markers in clothing or tools."
            ),
            "prompt_b": (
                f"A professional studio portrait of an older {concept}, isolated on a plain white background, "
                f"centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
                f"unambiguous profession markers in clothing or tools."
            ),
        },
    ]


def save_gemini_axis_outputs(concept: str, prompt_text: str, raw_text: str, parsed: dict, full_response: dict):
    out_dir = Path(AXES_ROOT) / slugify(concept)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "gemini_prompt.txt", "w", encoding="utf-8") as f:
        f.write(prompt_text)

    with open(out_dir / "gemini_raw_response.txt", "w", encoding="utf-8") as f:
        f.write(raw_text)

    with open(out_dir / "gemini_parsed_axes.json", "w", encoding="utf-8") as f:
        json.dump(parsed, f, indent=2, ensure_ascii=False)

    with open(out_dir / "gemini_full_response.json", "w", encoding="utf-8") as f:
        json.dump(full_response, f, indent=2, ensure_ascii=False)


def main():
    print(f"[SETUP] Professions: {PROFESSIONS}")
    print(f"[SETUP] Saving activations to: {ACTIVATIONS_ROOT}")
    print(f"[SETUP] Saving steering vectors to: {VECTORS_ROOT}")
    print(f"[SETUP] Saving Gemini axes to: {AXES_ROOT}")

    print("[SETUP] Loading diffusion pipeline...")
    pipe = init_pipeline()

    Path(OUTPUT_ROOT).mkdir(parents=True, exist_ok=True)
    Path(IMAGES_ROOT).mkdir(parents=True, exist_ok=True)
    Path(ACTIVATIONS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(VECTORS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(HEATMAPS_ROOT).mkdir(parents=True, exist_ok=True)
    Path(AXES_ROOT).mkdir(parents=True, exist_ok=True)

    results_csv_path = f"{OUTPUT_ROOT}/bias_analysis.csv"
    with open(results_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profession",
            "axis_name",
            "layer",
            "step",
            "continuous_bias",
            "mean_based_bias",
            "pole_a",
            "pole_b",
            "axis_source",
        ])

    text_bias_csv_path = f"{OUTPUT_ROOT}/text_bias_analysis.csv"
    with open(text_bias_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "profession",
            "axis_name",
            "text_embedding_bias",
            "dist_pole_a",
            "dist_pole_b",
            "axis_source",
        ])

    gemini_api_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_api_key:
        print(f"[SETUP] Gemini enabled: {GEMINI_MODEL}")
    else:
        print("[SETUP] GEMINI_API_KEY not found. Using fallback axes.")

    for profession in PROFESSIONS:
        profession_slug = slugify(profession)
        print(f"\n[PROFESSION] {profession}")

        try:
            if gemini_api_key:
                prompt_text, axes, raw_text, parsed_axes, full_response = request_gemini_axes(
                    profession,
                    gemini_api_key,
                )
                save_gemini_axis_outputs(
                    concept=profession,
                    prompt_text=prompt_text,
                    raw_text=raw_text,
                    parsed=parsed_axes,
                    full_response=full_response,
                )
                axis_source = "gemini"
            else:
                axes = fallback_axes(profession)
                axis_source = "fallback"
        except Exception as err:
            print(f"  - Gemini axis discovery failed: {repr(err)}")
            print("  - Using fallback axes.")
            axes = fallback_axes(profession)
            axis_source = "fallback"

        for axis in axes:
            axis_name = axis.get("axis_name", f"axis_{axis.get('rank', 'x')}")
            pole_a = axis.get("pole_a", "pole_a")
            pole_b = axis.get("pole_b", "pole_b")
            prompt_a = axis["prompt_a"]
            prompt_b = axis["prompt_b"]
            base_prompt = axis["base_prompt"]

            axis_slug = slugify(axis_name)

            print(f"\n  [AXIS] {axis_name}")
            print(f"    pole_a: {pole_a}")
            print(f"    pole_b: {pole_b}")
            print(f"    prompt_a: {prompt_a}")
            print(f"    prompt_b: {prompt_b}")
            print(f"    base_prompt: {base_prompt}")

            print("    - Analyzing text embeddings...")
            a_emb = get_sdxl_text_embeddings(pipe, prompt_a)
            b_emb = get_sdxl_text_embeddings(pipe, prompt_b)
            n_emb = get_sdxl_text_embeddings(pipe, base_prompt)

            text_bias, d_a, d_b = compute_static_text_bias(a_emb, b_emb, n_emb)
            print_bias_bar(f"{profession} | {axis_name}", text_bias)

            with open(text_bias_csv_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    profession,
                    axis_name,
                    f"{text_bias:.4f}",
                    f"{d_a:.4f}",
                    f"{d_b:.4f}",
                    axis_source,
                ])

            pole_a_traj = collect_prompt_trajectories(
                pipe,
                prompt_a,
                samples=AXIS_SAMPLES_PER_GROUP,
                layer_keys=TARGET_LAYER_KEYS,
                image_dir=f"{IMAGES_ROOT}/{profession_slug}/{axis_slug}/pole_a",
                activation_dir=f"{ACTIVATIONS_ROOT}/{profession_slug}/{axis_slug}/pole_a",
                phase_name=f"{profession} | {axis_name} | A",
            )

            pole_b_traj = collect_prompt_trajectories(
                pipe,
                prompt_b,
                samples=AXIS_SAMPLES_PER_GROUP,
                layer_keys=TARGET_LAYER_KEYS,
                image_dir=f"{IMAGES_ROOT}/{profession_slug}/{axis_slug}/pole_b",
                activation_dir=f"{ACTIVATIONS_ROOT}/{profession_slug}/{axis_slug}/pole_b",
                phase_name=f"{profession} | {axis_name} | B",
            )

            base_traj = collect_prompt_trajectories(
                pipe,
                base_prompt,
                samples=BASE_SAMPLES_PER_AXIS,
                layer_keys=TARGET_LAYER_KEYS,
                image_dir=f"{IMAGES_ROOT}/{profession_slug}/{axis_slug}/base",
                activation_dir=f"{ACTIVATIONS_ROOT}/{profession_slug}/{axis_slug}/base",
                phase_name=f"{profession} | {axis_name} | BASE",
            )

            analysis_results = compute_axis_bias_analysis(
                pole_a_traj,
                pole_b_traj,
                base_traj,
                TARGET_LAYER_KEYS,
            )

            heatmap_path = f"{HEATMAPS_ROOT}/{profession_slug}_{axis_slug}.png"
            plot_bias_heatmap(profession, axis_name, analysis_results, heatmap_path)

            save_steering_vectors(
                profession=profession,
                axis_name=axis_name,
                analysis_results=analysis_results,
                output_dir=VECTORS_ROOT,
            )

            with open(results_csv_path, "a", newline="") as f:
                writer = csv.writer(f)
                for res in analysis_results:
                    writer.writerow([
                        profession,
                        axis_name,
                        res["layer"],
                        res["step"],
                        f"{res['continuous']:.4f}",
                        f"{res['mean_based']:.4f}",
                        pole_a,
                        pole_b,
                        axis_source,
                    ])

            avg_bias = sum(r["continuous"] for r in analysis_results) / len(analysis_results)
            print("    [BIAS SUMMARY]")
            print_bias_bar(f"{profession} | {axis_name}", avg_bias)

            max_bias_entry = max(analysis_results, key=lambda x: abs(x["continuous"]))
            print(
                f"    Max Bias: {max_bias_entry['continuous']:.4f} "
                f"at {max_bias_entry['layer']} step {max_bias_entry['step']}"
            )

    print(f"\n[DONE] Bias analysis saved to {results_csv_path}")
    print(f"[DONE] Text bias analysis saved to {text_bias_csv_path}")
    print(f"[DONE] Steering vectors saved to {VECTORS_ROOT}")
    print(f"[DONE] Gemini axes saved to {AXES_ROOT}")
    print(f"[DONE] Heatmaps saved to {HEATMAPS_ROOT}")


if __name__ == "__main__":
    main()