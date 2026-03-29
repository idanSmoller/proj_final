#!/usr/bin/env python3
"""
Text Embedding Projection Experiment

This script projects CLIP text embeddings onto a hyperplane orthogonal to the
gender axis (male_prompt - female_prompt embedding) during image generation.
The projected embeddings are used in cross-attention at every diffusion step.

Unlike projection.py (which projects UNet activations), this script projects
the text prompt embeddings themselves before they enter the cross-attention layers.
"""

import gc
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from diffusers import DiffusionPipeline
from PIL import Image


MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
DEVICE = "cuda"
INFERENCE_STEPS = 30
GUIDANCE_SCALE = 5.0
SEED = 42

# Directories
OUTPUT_ROOT = "projection_text_results"

# Experiment configurations
PROFESSIONS_TO_TEST = ["Nurse", "Programmer", "Doctor"]

# Strength values control hyperplane position
STRENGTH_VALUES = [
    1.0,    # Male center
    0.5,    # Closer to male
    0.0,    # Midpoint (equal distance)
    -0.5,   # Closer to female
    -1.0,   # Female center
]

# Projection scale factors (how aggressively to project)
SCALE_FACTORS = [
    1.0,    # Standard projection
    2.0,    # 2x projection
    5.0,    # 5x projection
    10.0,   # 10x projection
]

# Prompt templates
MALE_PROMPT_TEMPLATE = (
    "A professional studio portrait of a male {profession}, isolated on a plain white background, "
    "centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
    "unambiguous profession markers in clothing or tools."
)

FEMALE_PROMPT_TEMPLATE = (
    "A professional studio portrait of a female {profession}, isolated on a plain white background, "
    "centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
    "unambiguous profession markers in clothing or tools."
)

NEUTRAL_PROMPT_TEMPLATE = (
    "A professional studio portrait of a {profession}, isolated on a plain white background, "
    "centered composition, sharp focus on the subject, high detail, photorealistic lighting, "
    "unambiguous profession markers in clothing or tools."
)

NEGATIVE_PROMPT = "background, details, clutter, messy, text, watermark, low quality, distorted, deformed, ugly, blur, out of focus"

NUM_SAMPLES_PER_CONFIG = 4


def init_pipeline(model_id: str = MODEL_ID, device: str = DEVICE):
    """Initialize the diffusion pipeline with memory optimizations."""
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


def slugify(text: str, max_len: int = 60) -> str:
    """Convert text to filesystem-safe slug."""
    slug = "".join(char if char.isalnum() else "_" for char in text).strip("_").lower()
    return slug[:max_len]


def encode_prompt(pipe, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode text prompts using CLIP text encoder(s).

    For SDXL, this uses both text_encoder and text_encoder_2.
    Returns pooled and non-pooled embeddings.
    """
    # SDXL has two text encoders
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    prompt_embeds_list = []
    pooled_prompt_embeds_list = []

    for text_encoder, tokenizer in zip(text_encoders, tokenizers):
        # Tokenize
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        # Encode
        with torch.no_grad():
            prompt_embeds = text_encoder(
                text_input_ids.to(text_encoder.device),
                output_hidden_states=True,
            )

        # Get hidden states and pooled output
        pooled_prompt_embeds = prompt_embeds[0]  # Pooled output
        prompt_embeds = prompt_embeds.hidden_states[-2]  # Last hidden state

        prompt_embeds_list.append(prompt_embeds)
        pooled_prompt_embeds_list.append(pooled_prompt_embeds)

    # Concatenate embeddings from both encoders (SDXL specific)
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds_list[-1]  # Use second encoder's pooled

    # Handle negative prompt similarly if provided
    if negative_prompt:
        negative_embeds_list = []
        negative_pooled_list = []

        for text_encoder, tokenizer in zip(text_encoders, tokenizers):
            neg_inputs = tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            with torch.no_grad():
                neg_embeds = text_encoder(
                    neg_inputs.input_ids.to(text_encoder.device),
                    output_hidden_states=True,
                )

            negative_pooled_list.append(neg_embeds[0])
            negative_embeds_list.append(neg_embeds.hidden_states[-2])

        negative_prompt_embeds = torch.cat(negative_embeds_list, dim=-1)
        negative_pooled_embeds = negative_pooled_list[-1]
    else:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_embeds = torch.zeros_like(pooled_prompt_embeds)

    return (prompt_embeds, pooled_prompt_embeds,
            negative_prompt_embeds, negative_pooled_embeds)


def compute_gender_axis(pipe, profession: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute gender axis in text embedding space.

    Returns:
        male_embeds: Male prompt embeddings
        female_embeds: Female prompt embeddings
        male_pooled: Male pooled embeddings
        female_pooled: Female pooled embeddings
    """
    male_prompt = MALE_PROMPT_TEMPLATE.format(profession=profession)
    female_prompt = FEMALE_PROMPT_TEMPLATE.format(profession=profession)

    # Encode male prompt
    male_embeds, male_pooled, _, _ = encode_prompt(pipe, male_prompt, "")

    # Encode female prompt
    female_embeds, female_pooled, _, _ = encode_prompt(pipe, female_prompt, "")

    return male_embeds, female_embeds, male_pooled, female_pooled


def project_to_hyperplane(
    embedding: torch.Tensor,
    male_embedding: torch.Tensor,
    female_embedding: torch.Tensor,
    strength: float = 0.0,
) -> torch.Tensor:
    """
    Project text embedding onto hyperplane orthogonal to gender axis.

    The hyperplane is orthogonal to (male_embedding - female_embedding) and passes
    through a point determined by the strength parameter:
    - strength = -1: hyperplane passes through female_embedding
    - strength =  0: hyperplane passes through midpoint (equal distance)
    - strength = +1: hyperplane passes through male_embedding

    Args:
        embedding: Current text embedding [B, seq_len, dim] or [B, dim]
        male_embedding: Male prompt embedding (same shape)
        female_embedding: Female prompt embedding (same shape)
        strength: Controls hyperplane position along gender axis [-1, 1]

    Returns:
        Projected embedding with same shape as input
    """
    original_shape = embedding.shape

    # Flatten to [B, -1] for projection
    flat_embedding = embedding.flatten(start_dim=1)
    flat_male = male_embedding.flatten(start_dim=1)
    flat_female = female_embedding.flatten(start_dim=1)

    # Compute gender axis
    gender_axis = flat_male - flat_female  # [B, dim]

    # Compute hyperplane anchor point based on strength
    # t = 0 (strength=-1) → female_embedding
    # t = 0.5 (strength=0) → midpoint
    # t = 1 (strength=1) → male_embedding
    t = (strength + 1.0) / 2.0
    anchor_point = flat_female + t * gender_axis

    # Project onto hyperplane
    # v_proj = v - ((v - anchor) · axis / ||axis||^2) * axis
    axis_norm_sq = (gender_axis * gender_axis).sum(dim=-1, keepdim=True)

    # Avoid division by zero
    axis_norm_sq = torch.clamp(axis_norm_sq, min=1e-8)

    projection_coeff = ((flat_embedding - anchor_point) * gender_axis).sum(dim=-1, keepdim=True) / axis_norm_sq
    projected = flat_embedding - projection_coeff * gender_axis

    # Reshape back to original shape
    return projected.reshape(original_shape)


def generate_with_text_projection(
    pipe,
    profession: str,
    strength: float = 0.0,
    scale_factor: float = 1.0,
    num_samples: int = 1,
    seed: int = SEED,
    projection_steps: Optional[int] = None
) -> List[Image.Image]:
    """
    Generate images with text embedding projection.

    Args:
        pipe: Diffusion pipeline
        profession: Profession name
        strength: Hyperplane position control [-1=female, 0=midpoint, +1=male]
        scale_factor: Amplification factor for projection (>1 = more aggressive)
        num_samples: Number of images to generate
        seed: Random seed
        projection_steps: Number of initial diffusion steps to use projected embeddings.
                         If None, use projected embeddings for all steps (default behavior).
                         If set to N, use projected embeddings for first N steps, then switch to original.

    Returns:
        List of generated PIL images
    """
    # Get gender axis in text embedding space
    male_embeds, female_embeds, male_pooled, female_pooled = compute_gender_axis(pipe, profession)

    # Encode neutral prompt
    neutral_prompt = NEUTRAL_PROMPT_TEMPLATE.format(profession=profession)
    neutral_embeds, neutral_pooled, neg_embeds, neg_pooled = encode_prompt(
        pipe, neutral_prompt, NEGATIVE_PROMPT
    )

    # Project neutral embeddings
    projected_embeds = project_to_hyperplane(
        neutral_embeds,
        male_embeds,
        female_embeds,
        strength=strength
    )

    projected_pooled = project_to_hyperplane(
        neutral_pooled,
        male_pooled,
        female_pooled,
        strength=strength
    )

    # Apply scale factor (amplify the projection effect)
    if scale_factor != 1.0:
        correction_embeds = projected_embeds - neutral_embeds
        projected_embeds = neutral_embeds + scale_factor * correction_embeds

        correction_pooled = projected_pooled - neutral_pooled
        projected_pooled = neutral_pooled + scale_factor * correction_pooled

    # Generate images
    images = []
    for i in range(num_samples):
        generator = torch.Generator(device=DEVICE).manual_seed(seed + i)

        if projection_steps is None:
            # Use projected embeddings for all steps (original behavior)
            result = pipe(
                prompt_embeds=projected_embeds,
                pooled_prompt_embeds=projected_pooled,
                negative_prompt_embeds=neg_embeds,
                negative_pooled_prompt_embeds=neg_pooled,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator
            )
            images.extend(result.images)
        else:
            # Use timestep-conditional projection
            result = generate_with_timestep_conditional_projection(
                pipe=pipe,
                projected_embeds=projected_embeds,
                projected_pooled=projected_pooled,
                original_embeds=neutral_embeds,
                original_pooled=neutral_pooled,
                neg_embeds=neg_embeds,
                neg_pooled=neg_pooled,
                projection_steps=projection_steps,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator
            )
            images.extend(result.images)

    return images


def generate_with_timestep_conditional_projection(
    pipe,
    projected_embeds: torch.Tensor,
    projected_pooled: torch.Tensor,
    original_embeds: torch.Tensor,
    original_pooled: torch.Tensor,
    neg_embeds: torch.Tensor,
    neg_pooled: torch.Tensor,
    projection_steps: int,
    num_inference_steps: int,
    guidance_scale: float,
    generator: torch.Generator
) -> Image.Image:
    """
    Generate image with conditional projection based on timestep.

    Uses projected embeddings for the first N steps, then switches to original embeddings.
    """
    import types

    # Store state for the hook
    class ProjectionState:
        def __init__(self):
            self.current_step = 0
            self.projection_steps = projection_steps
            self.projected_embeds = projected_embeds
            self.projected_pooled = projected_pooled
            self.original_embeds = original_embeds
            self.original_pooled = original_pooled
            self.switched = False

    state = ProjectionState()

    # Create a callback to track steps and switch embeddings
    def callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
        state.current_step = step_index

        # Switch embeddings after projection_steps
        if step_index >= state.projection_steps and not state.switched:
            print(f"    [SWITCH] Switching to original embeddings at step {step_index}")
            state.switched = True
            # Note: We can't directly modify the embeddings in the callback
            # We need to use a different approach (see below)

        return callback_kwargs

    # Alternative approach: Hook into the UNet's encoder_hidden_states
    original_forward = pipe.unet.forward

    def hooked_forward(sample, timestep, encoder_hidden_states, *args, **kwargs):
        # Determine which embeddings to use based on current step
        if state.current_step < state.projection_steps:
            # Use projected embeddings
            current_embeds = state.projected_embeds
            # Handle classifier-free guidance (concatenated embeddings)
            if encoder_hidden_states.shape[0] == 2 * current_embeds.shape[0]:
                encoder_hidden_states = torch.cat([neg_embeds, current_embeds])
        else:
            # Use original embeddings
            current_embeds = state.original_embeds
            if encoder_hidden_states.shape[0] == 2 * current_embeds.shape[0]:
                encoder_hidden_states = torch.cat([neg_embeds, current_embeds])

        return original_forward(sample, timestep, encoder_hidden_states, *args, **kwargs)

    # Temporarily replace the forward method
    pipe.unet.forward = types.MethodType(hooked_forward, pipe.unet)

    try:
        # Start with projected embeddings for the first steps
        result = pipe(
            prompt_embeds=projected_embeds,
            pooled_prompt_embeds=projected_pooled,
            negative_prompt_embeds=neg_embeds,
            negative_pooled_prompt_embeds=neg_pooled,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            callback_on_step_end=callback_on_step_end
        )
    finally:
        # Restore original forward method
        pipe.unet.forward = original_forward

    return result


def run_text_projection_experiment(
    pipe,
    profession: str,
    output_dir: Path,
    strength: float = 0.0,
    scale_factor: float = 1.0,
    num_samples: int = NUM_SAMPLES_PER_CONFIG,
    projection_steps: Optional[int] = None
):
    """
    Run a single text projection experiment configuration.

    Args:
        pipe: Diffusion pipeline
        profession: Profession name
        output_dir: Output directory for results
        strength: Hyperplane position control [-1=female, 0=midpoint, +1=male]
        scale_factor: Amplification factor for projection effect
        num_samples: Number of samples to generate
        projection_steps: Number of initial steps to use projection (None = all steps)
    """
    profession_slug = slugify(profession)
    strength_str = f"strength_{strength:.2f}".replace(".", "p").replace("-", "neg")
    scale_str = f"scale_{scale_factor:.1f}".replace(".", "p")

    if projection_steps is not None:
        steps_str = f"_first{projection_steps}steps"
        config_name = f"{profession_slug}__text_projection__{strength_str}__{scale_str}{steps_str}"
    else:
        config_name = f"{profession_slug}__text_projection__{strength_str}__{scale_str}"

    config_dir = output_dir / config_name
    config_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Running: {config_name}")

    # Generate images with text projection
    images = generate_with_text_projection(
        pipe=pipe,
        profession=profession,
        strength=strength,
        scale_factor=scale_factor,
        num_samples=num_samples,
        seed=SEED,
        projection_steps=projection_steps
    )

    # Save images
    for idx, img in enumerate(images):
        img_path = config_dir / f"sample_{idx+1}.png"
        img.save(img_path)

    # Save configuration metadata
    projection_desc = (
        f"Projected embeddings used for first {projection_steps} steps, "
        f"then switched to original embeddings."
        if projection_steps is not None
        else "Affects all diffusion steps."
    )

    metadata = {
        "profession": profession,
        "projection_type": "text_embedding",
        "prompt": NEUTRAL_PROMPT_TEMPLATE.format(profession=profession),
        "strength": strength,
        "scale_factor": scale_factor,
        "num_samples": num_samples,
        "inference_steps": INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "seed": SEED,
        "projection_steps": projection_steps,
        "description": (
            "Text embedding projection: Projects CLIP embeddings onto hyperplane "
            f"orthogonal to gender axis before cross-attention. {projection_desc}"
        )
    }

    metadata_path = config_dir / "config.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {len(images)} images to {config_dir}")


def main():
    """Main experiment runner."""
    print("[SETUP] Text Embedding Projection Experiment")
    print(f"[SETUP] Output directory: {OUTPUT_ROOT}")
    print(f"[SETUP] Professions: {PROFESSIONS_TO_TEST}")
    print(f"[SETUP] Strength values: {STRENGTH_VALUES}")
    print(f"[SETUP] Scale factors: {SCALE_FACTORS}")

    # Create output directory
    output_root = Path(OUTPUT_ROOT)
    output_root.mkdir(parents=True, exist_ok=True)

    # Initialize pipeline
    print("\n[SETUP] Loading diffusion pipeline...")
    pipe = init_pipeline()

    # Run experiments for each profession
    for profession in PROFESSIONS_TO_TEST:
        print(f"\n[PROFESSION] {profession}")

        try:
            # Run experiments for each configuration
            for strength in STRENGTH_VALUES:
                for scale_factor in SCALE_FACTORS:
                    run_text_projection_experiment(
                        pipe=pipe,
                        profession=profession,
                        output_dir=output_root,
                        strength=strength,
                        scale_factor=scale_factor,
                        num_samples=NUM_SAMPLES_PER_CONFIG
                    )

        except Exception as e:
            print(f"  [ERROR] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n[DONE] All experiments complete!")
    print(f"[DONE] Results saved to {OUTPUT_ROOT}")

    # Generate summary
    summary_path = output_root / "experiment_summary.json"
    summary = {
        "experiment_type": "text_embedding_projection",
        "professions": PROFESSIONS_TO_TEST,
        "strength_values": STRENGTH_VALUES,
        "scale_factors": SCALE_FACTORS,
        "samples_per_config": NUM_SAMPLES_PER_CONFIG,
        "total_configs": len(PROFESSIONS_TO_TEST) * len(STRENGTH_VALUES) * len(SCALE_FACTORS),
        "description": (
            "Projects CLIP text embeddings onto hyperplane orthogonal to gender axis. "
            "The projected embeddings are used in cross-attention at every diffusion step."
        )
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
