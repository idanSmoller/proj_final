#!/usr/bin/env python3
"""
Debiased Contextual Scrubbing and Vector Injection

This script implements a two-stage pipeline:
1. Text Bias Mitigation: Projects the neutral prompt's text embedding onto an 
   orthogonal hyperplane to mathematically remove inherent gender bias.
2. Activation Steering: Injects targeted, robustly discovered UNet activation 
   vectors to explicitly steer the generation towards masculine or feminine traits.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image


# =========================================================
# Logging / utilities
# =========================================================

def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def get_default_dtype(device: str) -> torch.dtype:
    return torch.float16 if device == "cuda" else torch.float32

def normalize_vec(x: torch.Tensor) -> torch.Tensor:
    x = x.detach().float()
    n = x.norm(p=2)
    if n.item() == 0:
        return x
    return x / (n + 1e-8)

def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower().replace(" ", "_")
    safe = "".join(ch for ch in text if ch.isalnum() or ch in "_-")
    return safe[:max_len]

def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)

def torch_save(path: str | Path, tensor: torch.Tensor) -> None:
    ensure_dir(Path(path).parent)
    torch.save(tensor.detach().cpu(), str(path))

def save_json(path: str | Path, obj: Dict) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# =========================================================
# Spatial mask & Hook Utils
# =========================================================

def get_spatial_mask(h: int, w: int, device: torch.device, dtype: torch.dtype, sigma: float) -> torch.Tensor:
    if sigma <= 0.0:
        return torch.ones((1, 1, h, w), device=device, dtype=dtype)
    y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    mask = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))
    return mask.view(1, 1, h, w)

def masked_channel_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor.shape)}")
    if mask.shape != (1, 1, tensor.shape[2], tensor.shape[3]):
        raise ValueError(f"Mask shape {tuple(mask.shape)} incompatible with tensor shape {tuple(tensor.shape)}")
    weighted = tensor * mask
    denom = mask.sum() * tensor.shape[0]
    if denom.item() == 0:
        raise ValueError("Masked mean denominator is zero")
    return weighted.sum(dim=(0, 2, 3)) / denom


# =========================================================
# Text Embedding Projection Utils
# =========================================================

def encode_prompt(pipe: StableDiffusionXLPipeline, prompt: str, negative_prompt: str = "") -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]

    prompt_embeds_list = []
    pooled_prompt_embeds_list = []

    for text_encoder, tokenizer in zip(text_encoders, tokenizers):
        text_inputs = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
        with torch.no_grad():
            prompt_embeds = text_encoder(text_inputs.input_ids.to(text_encoder.device), output_hidden_states=True)
        pooled_prompt_embeds_list.append(prompt_embeds[0])
        prompt_embeds_list.append(prompt_embeds.hidden_states[-2])

    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled_prompt_embeds_list[-1]

    if negative_prompt:
        negative_embeds_list = []
        negative_pooled_list = []
        for text_encoder, tokenizer in zip(text_encoders, tokenizers):
            neg_inputs = tokenizer(negative_prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
            with torch.no_grad():
                neg_embeds = text_encoder(neg_inputs.input_ids.to(text_encoder.device), output_hidden_states=True)
            negative_pooled_list.append(neg_embeds[0])
            negative_embeds_list.append(neg_embeds.hidden_states[-2])
        negative_prompt_embeds = torch.cat(negative_embeds_list, dim=-1)
        negative_pooled_embeds = negative_pooled_list[-1]
    else:
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_embeds = torch.zeros_like(pooled_prompt_embeds)

    return prompt_embeds, pooled_prompt_embeds, negative_prompt_embeds, negative_pooled_embeds

def project_to_hyperplane(embedding: torch.Tensor, male_embedding: torch.Tensor, female_embedding: torch.Tensor, strength: float = 0.0) -> torch.Tensor:
    original_shape = embedding.shape
    flat_embedding = embedding.flatten(start_dim=1)
    flat_male = male_embedding.flatten(start_dim=1)
    flat_female = female_embedding.flatten(start_dim=1)

    gender_axis = flat_male - flat_female
    t = (strength + 1.0) / 2.0
    anchor_point = flat_female + t * gender_axis

    axis_norm_sq = (gender_axis * gender_axis).sum(dim=-1, keepdim=True)
    axis_norm_sq = torch.clamp(axis_norm_sq, min=1e-8)

    projection_coeff = ((flat_embedding - anchor_point) * gender_axis).sum(dim=-1, keepdim=True) / axis_norm_sq
    projected = flat_embedding - projection_coeff * gender_axis

    return projected.reshape(original_shape)


# =========================================================
# Configs
# =========================================================

@dataclass
class CaptureConfig:
    module_path: str = "unet.down_blocks.2"
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    height: int = 1024
    width: int = 1024
    mask_sigma: float = 0.4
    discovery_start_step: int = 0
    discovery_end_step: Optional[int] = 12
    save_all_timestep_vectors: bool = True

@dataclass
class InjectorConfig:
    layer_key: str = "down_2"
    strength: float = 1.0
    cfg_target: str = "cond"
    normalize: str = "rms"
    start_step: int = 0
    end_step: Optional[int] = 12
    schedule: str = "flat"
    mask_sigma: float = 0.4
    log_every: int = 0

@dataclass
class TraceResult:
    prompt: str
    seed: int
    per_step_vectors: List[torch.Tensor]
    used_step_indices: List[int]


# =========================================================
# Schedule helpers
# =========================================================

def schedule_multiplier(
    step_idx: int,
    start_step: int,
    end_step: Optional[int],
    mode: str,
) -> float:
    if step_idx < start_step:
        return 0.0

    if end_step is not None and step_idx >= end_step:
        return 0.0

    if mode == "flat":
        return 1.0

    if end_step is None:
        return 1.0

    span = max(end_step - start_step, 1)
    t = (step_idx - start_step) / span
    t = min(max(t, 0.0), 1.0)

    if mode == "linear_decay":
        return 1.0 - t
    if mode == "cosine_decay":
        return 0.5 * (1.0 + math.cos(math.pi * t))

    raise ValueError(f"Unknown schedule mode: {mode}")


# =========================================================
# Activation Injector
# =========================================================

class ActivationInjector:
    def __init__(self, cfg: InjectorConfig, direction: torch.Tensor):
        self.cfg = cfg
        self.direction = normalize_vec(direction.detach().float().cpu())
        self.current_step = 0
        self.handle = None
        self.target_module = None

    def install(self, pipe: StableDiffusionXLPipeline) -> "ActivationInjector":
        unet = pipe.unet
        key = self.cfg.layer_key

        if key == "mid_block":
            self.target_module = unet.mid_block
        elif key.startswith("down_"):
            idx = int(key.split("_")[1])
            self.target_module = unet.down_blocks[idx]
        elif key.startswith("up_"):
            idx = int(key.split("_")[1])
            self.target_module = unet.up_blocks[idx]
        else:
            raise ValueError(f"Unknown layer_key: {key}")

        self.handle = self.target_module.register_forward_hook(self._hook_fn)
        self.current_step = 0
        return self

    def uninstall(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def _make_delta(self, target_tensor: torch.Tensor) -> torch.Tensor:
        direction = self.direction.to(target_tensor.device, dtype=target_tensor.dtype)

        if direction.numel() != target_tensor.shape[1]:
            raise ValueError(
                f"Direction length {direction.numel()} does not match target channels {target_tensor.shape[1]}"
            )

        update_spatial = direction.view(1, -1, 1, 1)

        if self.cfg.normalize == "rms":
            sample_scale = (
                target_tensor.pow(2)
                .mean(dim=(1, 2, 3), keepdim=True)
                .sqrt()
                .clamp_min(1e-8)
            )
            delta = self.cfg.strength * sample_scale * update_spatial
        elif self.cfg.normalize == "none":
            delta = self.cfg.strength * update_spatial
        else:
            raise ValueError(f"Unknown normalize mode: {self.cfg.normalize}")

        h, w = target_tensor.shape[2], target_tensor.shape[3]
        mask = get_spatial_mask(h, w, target_tensor.device, target_tensor.dtype, self.cfg.mask_sigma)
        delta = delta * mask
        return delta

    def _hook_fn(self, module, inputs, output):
        step_mult = schedule_multiplier(
            step_idx=self.current_step,
            start_step=self.cfg.start_step,
            end_step=self.cfg.end_step,
            mode=self.cfg.schedule,
        )

        if step_mult == 0.0:
            self.current_step += 1
            return output

        is_tuple = isinstance(output, tuple)
        tensor = output[0] if is_tuple else output

        if not torch.is_tensor(tensor) or tensor.dim() != 4:
            self.current_step += 1
            return output

        out = tensor.clone()

        if self.cfg.cfg_target == "cond":
            half = out.shape[0] // 2
            target = out[half:]
            delta = self._make_delta(target) * step_mult
            out[half:] = target + delta

        elif self.cfg.cfg_target == "uncond":
            half = out.shape[0] // 2
            target = out[:half]
            delta = self._make_delta(target) * step_mult
            out[:half] = target + delta

        elif self.cfg.cfg_target == "both":
            target = out
            delta = self._make_delta(target) * step_mult
            out = target + delta

        else:
            raise ValueError(f"Unknown cfg_target: {self.cfg.cfg_target}")

        self.current_step += 1
        return (out, *output[1:]) if is_tuple else out


# =========================================================
# Pipeline manager
# =========================================================

class SDXLPipelineManager:
    def __init__(self, model_id: str, enable_cpu_offload: bool):
        self.device = get_default_device()
        self.dtype = get_default_dtype(self.device)

        load_kwargs = {"torch_dtype": self.dtype, "use_safetensors": True}
        if self.device == "cuda" and self.dtype == torch.float16:
            load_kwargs["variant"] = "fp16"

        log(f"[init] Loading pipeline on {self.device}")
        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)
        self.pipe.enable_vae_slicing()

        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to(self.device)

        self.pipe.set_progress_bar_config(disable=True)

    def capture_trace(self, prompt: str, seed: int, cfg: CaptureConfig) -> TraceResult:
        parts = cfg.module_path.split(".")
        target_module = self.pipe
        for part in parts:
            target_module = target_module[int(part)] if part.isdigit() else getattr(target_module, part)

        per_step_vectors, used_step_indices = [], []
        hook_step = {"i": 0}

        def hook_fn(module, inputs, output):
            step_idx = hook_step["i"]
            hook_step["i"] += 1
            if step_idx < cfg.discovery_start_step or (cfg.discovery_end_step is not None and step_idx >= cfg.discovery_end_step):
                return
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor) or tensor.dim() != 4:
                return
            if tensor.shape[0] >= 2 and cfg.guidance_scale > 1.0:
                tensor = tensor[tensor.shape[0] // 2:]
            
            mask = get_spatial_mask(tensor.shape[2], tensor.shape[3], tensor.device, tensor.dtype, cfg.mask_sigma)
            vec = masked_channel_mean(tensor, mask)
            per_step_vectors.append(normalize_vec(vec.detach().cpu()))
            used_step_indices.append(step_idx)

        handle = target_module.register_forward_hook(hook_fn)
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self.pipe(prompt=prompt, num_inference_steps=cfg.num_inference_steps, guidance_scale=cfg.guidance_scale, height=cfg.height, width=cfg.width, generator=generator)
        finally:
            handle.remove()

        return TraceResult(prompt=prompt, seed=seed, per_step_vectors=per_step_vectors, used_step_indices=used_step_indices)

    @torch.inference_mode()
    def generate(self, cfg: CaptureConfig, seed: int, injector: Optional[ActivationInjector] = None, prompt: Optional[str] = None, prompt_embeds: Optional[torch.Tensor] = None, pooled_prompt_embeds: Optional[torch.Tensor] = None, negative_prompt_embeds: Optional[torch.Tensor] = None, negative_pooled_prompt_embeds: Optional[torch.Tensor] = None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if injector: injector.install(self.pipe)

        kwargs = {
            "num_inference_steps": cfg.num_inference_steps,
            "guidance_scale": cfg.guidance_scale,
            "height": cfg.height, "width": cfg.width,
            "generator": generator,
        }
        
        if prompt is not None:
            kwargs["prompt"] = prompt
        else:
            kwargs["prompt_embeds"] = prompt_embeds
            kwargs["pooled_prompt_embeds"] = pooled_prompt_embeds
            kwargs["negative_prompt_embeds"] = negative_prompt_embeds
            kwargs["negative_pooled_prompt_embeds"] = negative_pooled_prompt_embeds

        try:
            result = self.pipe(**kwargs)
            return result.images[0]
        finally:
            if injector: injector.uninstall()
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

def capture_robust_trace(manager: SDXLPipelineManager, prompt: str, seeds: List[int], cfg: CaptureConfig) -> TraceResult:
    all_step_vectors = []
    used_indices = None
    for s in seeds:
        trace = manager.capture_trace(prompt, s, cfg)
        if len(trace.per_step_vectors) == 0:
            raise RuntimeError(f"No activations captured for prompt='{prompt}'. Check module_path and discovery step range.")
        all_step_vectors.append(torch.stack(trace.per_step_vectors, dim=0))
        if used_indices is None: used_indices = trace.used_step_indices
    
    robust_mean_steps = torch.stack(all_step_vectors, dim=0).mean(dim=0)
    robust_step_list = [robust_mean_steps[i] for i in range(robust_mean_steps.shape[0])]
    return TraceResult(prompt=prompt, seed=seeds[0], per_step_vectors=robust_step_list, used_step_indices=used_indices)


# =========================================================
# Axis construction
# =========================================================

def mean_trace_vector(trace: TraceResult) -> torch.Tensor:
    return normalize_vec(torch.stack(trace.per_step_vectors, dim=0).mean(dim=0))


def build_independent_vectors(
    neutral_trace: TraceResult,
    male_trace: TraceResult,
    female_trace: TraceResult,
) -> Tuple[torch.Tensor, torch.Tensor]:
    neutral_mean = mean_trace_vector(neutral_trace)
    male_mean = mean_trace_vector(male_trace)
    female_mean = mean_trace_vector(female_trace)

    vec_masculine = normalize_vec(male_mean - neutral_mean)
    vec_feminine = normalize_vec(female_mean - neutral_mean)

    return vec_masculine, vec_feminine


# =========================================================
# CLI & Execution
# =========================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--enable-cpu-offload", action="store_true")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--professions", type=str, nargs="+", default= [
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
])
    p.add_argument("--num-discovery-seeds", type=int, default=3)
    p.add_argument("--min-strength", type=float, default=0.5)
    p.add_argument("--max-strength", type=float, default=2.0)
    p.add_argument("--num-strengths", type=int, default=4)
    p.add_argument("--output-dir", type=str, default="outputs/debiased_injection")
    return p.parse_args()

def main():
    args = parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)

    capture_cfg = CaptureConfig(num_inference_steps=args.steps)
    manager = SDXLPipelineManager(model_id=args.model_id, enable_cpu_offload=args.enable_cpu_offload)
    discovery_seeds = [args.seed + i for i in range(args.num_discovery_seeds)]
    neg_prompt = "background, details, clutter, messy, text, watermark, low quality, distorted, deformed, ugly, blur, out of focus"

    log(f"[start] Beginning Two-Stage Pipeline (Debias + Injection)")

    for profession in args.professions:
        prof_dir = Path(args.output_dir) / slugify(profession)
        ensure_dir(prof_dir)
        
        common = "studio lighting, face clearly visible, realistic profile picture, centered subject"
        prompts = {
            "neutral": f"A {common} image of a {profession}",
            "male": f"A {common} image of a male {profession}",
            "female": f"A {common} image of a female {profession}"
        }

        log(f"\n[stage 1] Calculating Orthogonal Projection for: {profession}")
        # Encode raw prompts
        m_emb, m_pool, _, _ = encode_prompt(manager.pipe, prompts["male"])
        f_emb, f_pool, _, _ = encode_prompt(manager.pipe, prompts["female"])
        n_emb, n_pool, neg_emb, neg_pool = encode_prompt(manager.pipe, prompts["neutral"], neg_prompt)

        # Project neutral embeddings to midpoint (strength=0.0)
        proj_emb = project_to_hyperplane(n_emb, m_emb, f_emb, strength=0.0)
        proj_pool = project_to_hyperplane(n_pool, m_pool, f_pool, strength=0.0)

        log(f"[stage 2] Discovering Activation Vectors (N={args.num_discovery_seeds} seeds)")
        n_trace = capture_robust_trace(manager, prompts["neutral"], discovery_seeds, capture_cfg)
        m_trace = capture_robust_trace(manager, prompts["male"], discovery_seeds, capture_cfg)
        f_trace = capture_robust_trace(manager, prompts["female"], discovery_seeds, capture_cfg)

        neutral_mean = mean_trace_vector(n_trace)
        male_mean = mean_trace_vector(m_trace)
        female_mean = mean_trace_vector(f_trace)
        vec_masculine, vec_feminine = build_independent_vectors(n_trace, m_trace, f_trace)

        torch_save(prof_dir / "neutral_mean.pt", neutral_mean)
        torch_save(prof_dir / "male_mean.pt", male_mean)
        torch_save(prof_dir / "female_mean.pt", female_mean)
        torch_save(prof_dir / "vec_masculine.pt", vec_masculine)
        torch_save(prof_dir / "vec_feminine.pt", vec_feminine)

        if capture_cfg.save_all_timestep_vectors:
            torch_save(prof_dir / "neutral_steps.pt", torch.stack(n_trace.per_step_vectors))
            torch_save(prof_dir / "male_steps.pt", torch.stack(m_trace.per_step_vectors))
            torch_save(prof_dir / "female_steps.pt", torch.stack(f_trace.per_step_vectors))

        save_json(
            prof_dir / "trace_metadata.json",
            {
                "profession": profession,
                "prompts": prompts,
                "discovery_seeds": discovery_seeds,
                "neutral_used_steps": n_trace.used_step_indices,
                "male_used_steps": m_trace.used_step_indices,
                "female_used_steps": f_trace.used_step_indices,
            },
        )

        log(f"[generate] Axis-debiased baseline for {profession}")
        baseline = manager.generate(cfg=capture_cfg, seed=args.seed, prompt_embeds=proj_emb, pooled_prompt_embeds=proj_pool, negative_prompt_embeds=neg_emb, negative_pooled_prompt_embeds=neg_pool)
        baseline.save(prof_dir / "unbiased_baseline.png")

        for strength in np.linspace(args.min_strength, args.max_strength, args.num_strengths):
            log(f"[generate] {profession} | MASCULINE | strength={strength:.2f}")
            inj_m = ActivationInjector(InjectorConfig(strength=strength), vec_masculine)
            img_m = manager.generate(cfg=capture_cfg, seed=args.seed, injector=inj_m, prompt_embeds=proj_emb, pooled_prompt_embeds=proj_pool, negative_prompt_embeds=neg_emb, negative_pooled_prompt_embeds=neg_pool)
            img_m.save(prof_dir / f"masculine_s{str(round(strength, 2)).replace('.', 'p')}.png")

            log(f"[generate] {profession} | FEMININE | strength={strength:.2f}")
            inj_f = ActivationInjector(InjectorConfig(strength=strength), vec_feminine)
            img_f = manager.generate(cfg=capture_cfg, seed=args.seed, injector=inj_f, prompt_embeds=proj_emb, pooled_prompt_embeds=proj_pool, negative_prompt_embeds=neg_emb, negative_pooled_prompt_embeds=neg_pool)
            img_f.save(prof_dir / f"feminine_s{str(round(strength, 2)).replace('.', 'p')}.png")

    log("[done] All debiased injections complete.")

if __name__ == "__main__":
    main()