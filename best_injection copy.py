# #!/usr/bin/env python3
# """
# Gender Bias Vector Injection - Repaired Version

# Key repairs:
# 1. Uses a single signed axis: male - female
# 2. Keeps neutral prompt for baseline only
# 3. Fixes cfg_target="both"
# 4. Uses an explicit discovery window over timesteps
# 5. Computes masked means correctly
# 6. Saves vectors and metadata
# 7. Fails loudly on channel mismatches
# """

# from __future__ import annotations

# import argparse
# import gc
# import json
# import math
# import os
# import random
# import time
# from dataclasses import asdict, dataclass
# from pathlib import Path
# from typing import Dict, List, Optional, Sequence, Tuple

# import torch
# from diffusers import StableDiffusionXLPipeline


# # =========================================================
# # Logging / utilities
# # =========================================================

# def log(msg: str) -> None:
#     now = time.strftime("%Y-%m-%d %H:%M:%S")
#     print(f"[{now}] {msg}", flush=True)


# def set_global_seed(seed: int) -> None:
#     random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)


# def get_default_device() -> str:
#     return "cuda" if torch.cuda.is_available() else "cpu"


# def get_default_dtype(device: str) -> torch.dtype:
#     return torch.float16 if device == "cuda" else torch.float32


# def normalize_vec(x: torch.Tensor) -> torch.Tensor:
#     x = x.detach().float()
#     n = x.norm(p=2)
#     if n.item() == 0:
#         return x
#     return x / (n + 1e-8)


# def slugify(text: str, max_len: int = 80) -> str:
#     text = text.strip().lower().replace(" ", "_")
#     safe = "".join(ch for ch in text if ch.isalnum() or ch in "_-")
#     return safe[:max_len]


# def ensure_dir(path: str | Path) -> None:
#     Path(path).mkdir(parents=True, exist_ok=True)


# def torch_save(path: str | Path, tensor: torch.Tensor) -> None:
#     ensure_dir(Path(path).parent)
#     torch.save(tensor.detach().cpu(), str(path))


# def save_json(path: str | Path, obj: Dict) -> None:
#     ensure_dir(Path(path).parent)
#     with open(path, "w", encoding="utf-8") as f:
#         json.dump(obj, f, indent=2, ensure_ascii=False)


# # =========================================================
# # Spatial mask
# # =========================================================

# def get_spatial_mask(
#     h: int,
#     w: int,
#     device: torch.device,
#     dtype: torch.dtype,
#     sigma: float,
# ) -> torch.Tensor:
#     """
#     Center-weighted Gaussian mask in latent space.
#     sigma <= 0 disables masking.
#     """
#     if sigma <= 0.0:
#         return torch.ones((1, 1, h, w), device=device, dtype=dtype)

#     y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
#     x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
#     gy, gx = torch.meshgrid(y, x, indexing="ij")
#     mask = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))
#     return mask.view(1, 1, h, w)


# def masked_channel_mean(
#     tensor: torch.Tensor,
#     mask: torch.Tensor,
# ) -> torch.Tensor:
#     """
#     tensor: [B, C, H, W]
#     mask:   [1, 1, H, W]
#     returns [C]
#     """
#     if tensor.dim() != 4:
#         raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor.shape)}")

#     if mask.shape != (1, 1, tensor.shape[2], tensor.shape[3]):
#         raise ValueError(
#             f"Mask shape {tuple(mask.shape)} incompatible with tensor shape {tuple(tensor.shape)}"
#         )

#     weighted = tensor * mask
#     denom = mask.sum() * tensor.shape[0]
#     if denom.item() == 0:
#         raise ValueError("Masked mean denominator is zero")
#     vec = weighted.sum(dim=(0, 2, 3)) / denom
#     return vec


# # =========================================================
# # Configs
# # =========================================================

# @dataclass
# class CaptureConfig:
#     module_path: str = "unet.down_blocks.2"
#     num_inference_steps: int = 30
#     guidance_scale: float = 5.0
#     height: int = 1024
#     width: int = 1024
#     mask_sigma: float = 0.4
#     discovery_start_step: int = 0
#     discovery_end_step: Optional[int] = 15
#     save_all_timestep_vectors: bool = True


# @dataclass
# class InjectorConfig:
#     layer_key: str = "down_2"
#     strength: float = 1.0
#     cfg_target: str = "cond"   # cond / uncond / both
#     normalize: str = "rms"     # none / rms
#     start_step: int = 0
#     end_step: Optional[int] = 15
#     schedule: str = "flat"     # flat / linear_decay / cosine_decay
#     mask_sigma: float = 0.4
#     log_every: int = 0


# @dataclass
# class TraceResult:
#     prompt: str
#     seed: int
#     per_step_vectors: List[torch.Tensor]
#     used_step_indices: List[int]


# # =========================================================
# # Schedule helpers
# # =========================================================

# def schedule_multiplier(
#     step_idx: int,
#     start_step: int,
#     end_step: Optional[int],
#     mode: str,
# ) -> float:
#     if step_idx < start_step:
#         return 0.0

#     if end_step is not None and step_idx >= end_step:
#         return 0.0

#     if mode == "flat":
#         return 1.0

#     if end_step is None:
#         # Cannot decay meaningfully without an end bound
#         return 1.0

#     span = max(end_step - start_step, 1)
#     t = (step_idx - start_step) / span
#     t = min(max(t, 0.0), 1.0)

#     if mode == "linear_decay":
#         return 1.0 - t
#     if mode == "cosine_decay":
#         return 0.5 * (1.0 + math.cos(math.pi * t))

#     raise ValueError(f"Unknown schedule mode: {mode}")


# # =========================================================
# # Activation Injector
# # =========================================================

# class ActivationInjector:
#     def __init__(self, cfg: InjectorConfig, direction: torch.Tensor):
#         self.cfg = cfg
#         self.direction = normalize_vec(direction.detach().float().cpu())
#         self.current_step = 0
#         self.handle = None
#         self.target_module = None

#     def install(self, pipe: StableDiffusionXLPipeline) -> "ActivationInjector":
#         unet = pipe.unet
#         key = self.cfg.layer_key

#         if key == "mid_block":
#             self.target_module = unet.mid_block
#         elif key.startswith("down_"):
#             idx = int(key.split("_")[1])
#             self.target_module = unet.down_blocks[idx]
#         elif key.startswith("up_"):
#             idx = int(key.split("_")[1])
#             self.target_module = unet.up_blocks[idx]
#         else:
#             raise ValueError(f"Unknown layer_key: {key}")

#         self.handle = self.target_module.register_forward_hook(self._hook_fn)
#         self.current_step = 0
#         return self

#     def uninstall(self) -> None:
#         if self.handle is not None:
#             self.handle.remove()
#             self.handle = None

#     def _make_delta(self, target_tensor: torch.Tensor) -> torch.Tensor:
#         """
#         target_tensor: [B, C, H, W]
#         """
#         direction = self.direction.to(target_tensor.device, dtype=target_tensor.dtype)

#         if direction.numel() != target_tensor.shape[1]:
#             raise ValueError(
#                 f"Direction length {direction.numel()} does not match target channels {target_tensor.shape[1]}"
#             )

#         update_spatial = direction.view(1, -1, 1, 1)

#         if self.cfg.normalize == "rms":
#             sample_scale = (
#                 target_tensor.pow(2)
#                 .mean(dim=(1, 2, 3), keepdim=True)
#                 .sqrt()
#                 .clamp_min(1e-8)
#             )
#             delta = self.cfg.strength * sample_scale * update_spatial
#         elif self.cfg.normalize == "none":
#             delta = self.cfg.strength * update_spatial
#         else:
#             raise ValueError(f"Unknown normalize mode: {self.cfg.normalize}")

#         h, w = target_tensor.shape[2], target_tensor.shape[3]
#         mask = get_spatial_mask(h, w, target_tensor.device, target_tensor.dtype, self.cfg.mask_sigma)
#         delta = delta * mask
#         return delta

#     def _hook_fn(self, module, inputs, output):
#         step_mult = schedule_multiplier(
#             step_idx=self.current_step,
#             start_step=self.cfg.start_step,
#             end_step=self.cfg.end_step,
#             mode=self.cfg.schedule,
#         )

#         if step_mult == 0.0:
#             self.current_step += 1
#             return output

#         is_tuple = isinstance(output, tuple)
#         tensor = output[0] if is_tuple else output

#         if not torch.is_tensor(tensor) or tensor.dim() != 4:
#             self.current_step += 1
#             return output

#         out = tensor.clone()

#         if self.cfg.cfg_target == "cond":
#             half = out.shape[0] // 2
#             target = out[half:]
#             delta = self._make_delta(target) * step_mult
#             out[half:] = target + delta

#         elif self.cfg.cfg_target == "uncond":
#             half = out.shape[0] // 2
#             target = out[:half]
#             delta = self._make_delta(target) * step_mult
#             out[:half] = target + delta

#         elif self.cfg.cfg_target == "both":
#             target = out
#             delta = self._make_delta(target) * step_mult
#             out = target + delta

#         else:
#             raise ValueError(f"Unknown cfg_target: {self.cfg.cfg_target}")

#         if self.cfg.log_every > 0 and self.current_step % self.cfg.log_every == 0:
#             with torch.no_grad():
#                 base_rms = tensor.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
#                 delta_rms = (out - tensor).pow(2).mean(dim=(1, 2, 3)).sqrt()
#                 ratio = (delta_rms / base_rms).mean().item()
#                 log(f"[injector] step={self.current_step:02d} effective_delta_ratio={ratio:.4f}")

#         self.current_step += 1
#         return (out, *output[1:]) if is_tuple else out


# # =========================================================
# # Pipeline manager
# # =========================================================

# class SDXLPipelineManager:
#     def __init__(self, model_id: str, enable_cpu_offload: bool):
#         self.device = get_default_device()
#         self.dtype = get_default_dtype(self.device)

#         load_kwargs = {
#             "torch_dtype": self.dtype,
#             "use_safetensors": True,
#         }
#         if self.device == "cuda" and self.dtype == torch.float16:
#             load_kwargs["variant"] = "fp16"

#         log(f"[init] Loading pipeline on {self.device}")
#         self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)
#         self.pipe.enable_vae_slicing()

#         if enable_cpu_offload:
#             self.pipe.enable_model_cpu_offload()
#         else:
#             self.pipe.to(self.device)

#         self.pipe.set_progress_bar_config(disable=True)

#     def _resolve_target_module(self, module_path: str) -> torch.nn.Module:
#         parts = module_path.split(".")
#         obj = self.pipe
#         for part in parts:
#             if part.isdigit():
#                 obj = obj[int(part)]
#             else:
#                 obj = getattr(obj, part)
#         return obj

#     @torch.inference_mode()
#     def capture_trace(self, prompt: str, seed: int, cfg: CaptureConfig) -> TraceResult:
#         target_module = self._resolve_target_module(cfg.module_path)
#         per_step_vectors: List[torch.Tensor] = []
#         used_step_indices: List[int] = []
#         hook_step = {"i": 0}

#         def hook_fn(module, inputs, output):
#             step_idx = hook_step["i"]
#             hook_step["i"] += 1

#             if step_idx < cfg.discovery_start_step:
#                 return

#             if cfg.discovery_end_step is not None and step_idx >= cfg.discovery_end_step:
#                 return

#             tensor = output[0] if isinstance(output, tuple) else output
#             if not torch.is_tensor(tensor) or tensor.dim() != 4:
#                 return

#             # For CFG batches, keep only conditional branch if guidance > 1
#             if tensor.shape[0] >= 2 and cfg.guidance_scale > 1.0:
#                 tensor = tensor[tensor.shape[0] // 2:]

#             h, w = tensor.shape[2], tensor.shape[3]
#             mask = get_spatial_mask(h, w, tensor.device, tensor.dtype, cfg.mask_sigma)
#             vec = masked_channel_mean(tensor, mask)
#             per_step_vectors.append(normalize_vec(vec.detach().cpu()))
#             used_step_indices.append(step_idx)

#         handle = target_module.register_forward_hook(hook_fn)
#         try:
#             generator = torch.Generator(device=self.device).manual_seed(seed)
#             _ = self.pipe(
#                 prompt=prompt,
#                 num_inference_steps=cfg.num_inference_steps,
#                 guidance_scale=cfg.guidance_scale,
#                 height=cfg.height,
#                 width=cfg.width,
#                 generator=generator,
#             )
#         finally:
#             handle.remove()

#         if len(per_step_vectors) == 0:
#             raise RuntimeError(
#                 "No timestep vectors were captured. Check module_path, discovery window, and hook target."
#             )

#         return TraceResult(
#             prompt=prompt,
#             seed=seed,
#             per_step_vectors=per_step_vectors,
#             used_step_indices=used_step_indices,
#         )

#     @torch.inference_mode()
#     def generate(
#         self,
#         prompt: str,
#         seed: int,
#         cfg: CaptureConfig,
#         injector: Optional[ActivationInjector] = None,
#     ):
#         generator = torch.Generator(device=self.device).manual_seed(seed)

#         if injector is not None:
#             injector.install(self.pipe)

#         try:
#             result = self.pipe(
#                 prompt=prompt,
#                 num_inference_steps=cfg.num_inference_steps,
#                 guidance_scale=cfg.guidance_scale,
#                 height=cfg.height,
#                 width=cfg.width,
#                 generator=generator,
#             )
#             img = result.images[0]
#             return img
#         finally:
#             if injector is not None:
#                 injector.uninstall()
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()


# # =========================================================
# # Axis construction
# # =========================================================

# def mean_trace_vector(trace: TraceResult) -> torch.Tensor:
#     return normalize_vec(torch.stack(trace.per_step_vectors, dim=0).mean(dim=0))


# def build_gender_axis(
#     male_trace: TraceResult,
#     female_trace: TraceResult,
# ) -> torch.Tensor:
#     male_mean = mean_trace_vector(male_trace)
#     female_mean = mean_trace_vector(female_trace)
#     axis = normalize_vec(male_mean - female_mean)
#     return axis


# # =========================================================
# # Prompt helpers
# # =========================================================

# def make_prompts(profession: str) -> Dict[str, str]:
#     common = "studio lighting, face clearly visible, realistic profile picture, centered subject"
#     return {
#         "neutral": f"A {common} image of a {profession}",
#         "male": f"A {common} image of a male {profession}",
#         "female": f"A {common} image of a female {profession}",
#     }


# # =========================================================
# # CLI
# # =========================================================

# def parse_args():
#     p = argparse.ArgumentParser()

#     p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
#     p.add_argument("--enable-cpu-offload", action="store_true")

#     p.add_argument("--module-path", type=str, default="unet.down_blocks.2")
#     p.add_argument(
#         "--inject-layer",
#         type=str,
#         default="down_2",
#         choices=["mid_block", "down_0", "down_1", "down_2", "up_0", "up_1", "up_2"],
#     )

#     p.add_argument("--steps", type=int, default=30)
#     p.add_argument("--guidance", type=float, default=5.0)
#     p.add_argument("--height", type=int, default=1024)
#     p.add_argument("--width", type=int, default=1024)
#     p.add_argument("--mask-sigma", type=float, default=0.4)

#     p.add_argument("--discovery-start-step", type=int, default=0)
#     p.add_argument("--discovery-end-step", type=int, default=15, help="Use -1 for no limit")

#     p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="cond")
#     p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
#     p.add_argument("--inject-start-step", type=int, default=0)
#     p.add_argument("--inject-end-step", type=int, default=15, help="Use -1 for no limit")
#     p.add_argument("--inject-schedule", type=str, choices=["flat", "linear_decay", "cosine_decay"], default="flat")
#     p.add_argument("--inject-log-every", type=int, default=0)

#     p.add_argument("--seed", type=int, default=42)
#     p.add_argument("--professions", type=str, nargs="+", default=["Nurse", "Doctor", "Programmer", "Kindergarten Teacher", "CEO", "Construction Worker", "Flight Attendant", "Scientist", "Firefighter", "Librarian"])
#     p.add_argument("--min-strength", type=float, default=0.5)
#     p.add_argument("--max-strength", type=float, default=2.0)
#     p.add_argument("--num-strengths", type=int, default=5)

#     p.add_argument("--output-dir", type=str, default="outputs/repaired_gender_axis_injection")

#     return p.parse_args()


# # =========================================================
# # Main
# # =========================================================

# def main():
#     args = parse_args()
#     set_global_seed(args.seed)
#     ensure_dir(args.output_dir)

#     capture_cfg = CaptureConfig(
#         module_path=args.module_path,
#         num_inference_steps=args.steps,
#         guidance_scale=args.guidance,
#         height=args.height,
#         width=args.width,
#         mask_sigma=args.mask_sigma,
#         discovery_start_step=args.discovery_start_step,
#         discovery_end_step=None if args.discovery_end_step < 0 else args.discovery_end_step,
#         save_all_timestep_vectors=True,
#     )

#     manager = SDXLPipelineManager(
#         model_id=args.model_id,
#         enable_cpu_offload=args.enable_cpu_offload,
#     )

#     global_meta = {
#         "seed": args.seed,
#         "model_id": args.model_id,
#         "capture_cfg": asdict(capture_cfg),
#         "inject_layer": args.inject_layer,
#         "inject_cfg_target": args.inject_cfg_target,
#         "inject_normalize": args.inject_normalize,
#         "inject_start_step": args.inject_start_step,
#         "inject_end_step": None if args.inject_end_step < 0 else args.inject_end_step,
#         "inject_schedule": args.inject_schedule,
#         "min_strength": args.min_strength,
#         "max_strength": args.max_strength,
#         "num_strengths": args.num_strengths,
#         "professions": args.professions,
#     }
#     save_json(Path(args.output_dir) / "run_config.json", global_meta)

#     log("[start] Beginning repaired gender-axis discovery + injection")

#     for profession in args.professions:
#         prof_slug = slugify(profession)
#         prof_dir = Path(args.output_dir) / prof_slug
#         ensure_dir(prof_dir)

#         prompts = make_prompts(profession)

#         log(f"[capture] Profession: {profession}")
#         log(f"  neutral: {prompts['neutral']}")
#         log(f"  male:    {prompts['male']}")
#         log(f"  female:  {prompts['female']}")

#         neutral_trace = manager.capture_trace(prompts["neutral"], args.seed, capture_cfg)
#         male_trace = manager.capture_trace(prompts["male"], args.seed, capture_cfg)
#         female_trace = manager.capture_trace(prompts["female"], args.seed, capture_cfg)

#         neutral_mean = mean_trace_vector(neutral_trace)
#         male_mean = mean_trace_vector(male_trace)
#         female_mean = mean_trace_vector(female_trace)
#         gender_axis = build_gender_axis(male_trace, female_trace)

#         # Save trace vectors
#         torch_save(prof_dir / "neutral_mean.pt", neutral_mean)
#         torch_save(prof_dir / "male_mean.pt", male_mean)
#         torch_save(prof_dir / "female_mean.pt", female_mean)
#         torch_save(prof_dir / "gender_axis.pt", gender_axis)

#         if capture_cfg.save_all_timestep_vectors:
#             torch_save(prof_dir / "neutral_steps.pt", torch.stack(neutral_trace.per_step_vectors))
#             torch_save(prof_dir / "male_steps.pt", torch.stack(male_trace.per_step_vectors))
#             torch_save(prof_dir / "female_steps.pt", torch.stack(female_trace.per_step_vectors))

#         save_json(
#             prof_dir / "trace_metadata.json",
#             {
#                 "profession": profession,
#                 "prompts": prompts,
#                 "neutral_used_steps": neutral_trace.used_step_indices,
#                 "male_used_steps": male_trace.used_step_indices,
#                 "female_used_steps": female_trace.used_step_indices,
#             },
#         )

#         # Baseline
#         log(f"[generate] Baseline for {profession}")
#         baseline = manager.generate(prompts["neutral"], args.seed, capture_cfg)
#         baseline.save(prof_dir / "baseline_neutral.png")

#         # Positive direction = more male
#         # Negative direction = more female
#         for strength in np.linspace(args.min_strength, args.max_strength, args.num_strengths):
#             log(f"[generate] {profession} | +axis | strength={strength}")
#             inj_cfg_pos = InjectorConfig(
#                 layer_key=args.inject_layer,
#                 strength=abs(strength),
#                 cfg_target=args.inject_cfg_target,
#                 normalize=args.inject_normalize,
#                 start_step=args.inject_start_step,
#                 end_step=None if args.inject_end_step < 0 else args.inject_end_step,
#                 schedule=args.inject_schedule,
#                 mask_sigma=args.mask_sigma,
#                 log_every=args.inject_log_every,
#             )
#             injector_pos = ActivationInjector(inj_cfg_pos, gender_axis)
#             img_pos = manager.generate(prompts["neutral"], args.seed, capture_cfg, injector=injector_pos)
#             img_pos.save(prof_dir / f"axis_pos_s{str(strength).replace('.', 'p')}.png")

#             log(f"[generate] {profession} | -axis | strength={strength}")
#             inj_cfg_neg = InjectorConfig(
#                 layer_key=args.inject_layer,
#                 strength=abs(strength),
#                 cfg_target=args.inject_cfg_target,
#                 normalize=args.inject_normalize,
#                 start_step=args.inject_start_step,
#                 end_step=None if args.inject_end_step < 0 else args.inject_end_step,
#                 schedule=args.inject_schedule,
#                 mask_sigma=args.mask_sigma,
#                 log_every=args.inject_log_every,
#             )
#             injector_neg = ActivationInjector(inj_cfg_neg, -gender_axis)
#             img_neg = manager.generate(prompts["neutral"], args.seed, capture_cfg, injector=injector_neg)
#             img_neg.save(prof_dir / f"axis_neg_s{str(strength).replace('.', 'p')}.png")

#     log("[done] All repaired injections complete.")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Gender Bias Vector Injection - Robust Multi-Seed Contextual Scrubbing

Key features:
1. Multi-Seed Averaging (High-Throughput Discovery) to eliminate noise artifacts.
2. Calculates independent vectors (M - N) and (F - N).
3. Applies targeted spatial masking during both capture and injection.
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
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from diffusers import StableDiffusionXLPipeline

# Import shared functions from identification module for reproducibility
from identification import (
    to_storable_tensor,
    energy_distance_1d,
    compute_bias_for_vectors as _compute_bias_for_vectors_base,
    get_target_module,
)


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
# Spatial mask
# =========================================================

def get_spatial_mask(
    h: int,
    w: int,
    device: torch.device,
    dtype: torch.dtype,
    sigma: float,
) -> torch.Tensor:
    """
    Center-weighted Gaussian mask in latent space.
    sigma <= 0 disables masking.
    """
    if sigma <= 0.0:
        return torch.ones((1, 1, h, w), device=device, dtype=dtype)

    y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    gy, gx = torch.meshgrid(y, x, indexing="ij")
    mask = torch.exp(-(gx**2 + gy**2) / (2 * sigma**2))
    return mask.view(1, 1, h, w)


def masked_channel_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    tensor: [B, C, H, W]
    mask:   [1, 1, H, W]
    returns [C]
    """
    if tensor.dim() != 4:
        raise ValueError(f"Expected 4D tensor, got shape {tuple(tensor.shape)}")

    if mask.shape != (1, 1, tensor.shape[2], tensor.shape[3]):
        raise ValueError(
            f"Mask shape {tuple(mask.shape)} incompatible with tensor shape {tuple(tensor.shape)}"
        )

    weighted = tensor * mask
    denom = mask.sum() * tensor.shape[0]
    if denom.item() == 0:
        raise ValueError("Masked mean denominator is zero")
    vec = weighted.sum(dim=(0, 2, 3)) / denom
    return vec


# =========================================================
# Configs
# =========================================================

@dataclass
class CaptureConfig:
    module_path: str = "unet.down_blocks.2"  # primary steering layer for injection
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    height: int = 1024
    width: int = 1024
    mask_sigma: float = 0.4
    discovery_start_step: int = 0
    discovery_end_step: Optional[int] = 12
    save_all_timestep_vectors: bool = True
    # Multi-layer extraction for heatmaps (union of steering + analysis layers)
    multi_layer_keys: List[str] = None  # e.g., ["mid_block", "up_0", "up_1", "up_2"]

    def __post_init__(self):
        if self.multi_layer_keys is None:
            self.multi_layer_keys = ["mid_block", "up_0", "up_1", "up_2"]


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


@dataclass
class MultiLayerTraceResult:
    """Holds activation traces from multiple UNet layers for heatmap analysis."""
    prompt: str
    seed: int
    # Dict mapping layer_key -> list of per-step vectors
    layer_traces: Dict[str, List[torch.Tensor]]
    used_step_indices: List[int]


@dataclass
class MultiLayerSampledTraceResult:
    """Holds activation traces from multiple UNet layers with individual samples preserved (not averaged)."""
    prompt: str
    seeds: List[int]
    # Dict mapping layer_key -> list of per-step tensors, each of shape [num_seeds, C]
    layer_traces_batched: Dict[str, List[torch.Tensor]]
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
# Pipeline manager & Robust Capture
# =========================================================

class SDXLPipelineManager:
    def __init__(self, model_id: str, enable_cpu_offload: bool):
        self.device = get_default_device()
        self.dtype = get_default_dtype(self.device)

        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
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

    def _resolve_target_module(self, module_path: str) -> torch.nn.Module:
        parts = module_path.split(".")
        obj = self.pipe
        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    @torch.inference_mode()
    def capture_trace(self, prompt: str, seed: int, cfg: CaptureConfig) -> TraceResult:
        target_module = self._resolve_target_module(cfg.module_path)
        per_step_vectors: List[torch.Tensor] = []
        used_step_indices: List[int] = []
        hook_step = {"i": 0}

        def hook_fn(module, inputs, output):
            step_idx = hook_step["i"]
            hook_step["i"] += 1

            if step_idx < cfg.discovery_start_step:
                return

            if cfg.discovery_end_step is not None and step_idx >= cfg.discovery_end_step:
                return

            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor) or tensor.dim() != 4:
                return

            if tensor.shape[0] >= 2 and cfg.guidance_scale > 1.0:
                tensor = tensor[tensor.shape[0] // 2:]

            h, w = tensor.shape[2], tensor.shape[3]
            mask = get_spatial_mask(h, w, tensor.device, tensor.dtype, cfg.mask_sigma)
            vec = masked_channel_mean(tensor, mask)
            per_step_vectors.append(normalize_vec(vec.detach().cpu()))
            used_step_indices.append(step_idx)

        handle = target_module.register_forward_hook(hook_fn)
        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            _ = self.pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
            )
        finally:
            handle.remove()

        if len(per_step_vectors) == 0:
            raise RuntimeError(
                "No timestep vectors were captured. Check module_path, discovery window, and hook target."
            )

        return TraceResult(
            prompt=prompt,
            seed=seed,
            per_step_vectors=per_step_vectors,
            used_step_indices=used_step_indices,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        cfg: CaptureConfig,
        injector: Optional[ActivationInjector] = None,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if injector is not None:
            injector.install(self.pipe)

        try:
            result = self.pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
            )
            img = result.images[0]
            return img
        finally:
            if injector is not None:
                injector.uninstall()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def generate_with_multi_layer_capture(
        self,
        prompt: str,
        seed: int,
        cfg: CaptureConfig,
        injector: Optional[ActivationInjector] = None,
    ) -> Tuple[any, MultiLayerTraceResult]:
        """
        Generate an image while simultaneously capturing multi-layer activations.
        Returns both the image and the multi-layer trace.
        """
        layer_keys = cfg.multi_layer_keys
        trajectories = {layer_key: [] for layer_key in layer_keys}
        handles = []
        hook_step = {"i": 0}

        def make_hook(layer_key):
            def hook(_, __, output):
                step_idx = hook_step["i"]

                if step_idx < cfg.discovery_start_step:
                    return
                if cfg.discovery_end_step is not None and step_idx >= cfg.discovery_end_step:
                    return

                if torch.is_tensor(output):
                    tensor = output
                elif isinstance(output, (tuple, list)):
                    tensors = [item for item in output if torch.is_tensor(item)]
                    if not tensors:
                        return
                    tensor = max(tensors, key=lambda t: t.numel())
                else:
                    return

                # For CFG batches, keep only conditional branch
                if tensor.shape[0] >= 2 and cfg.guidance_scale > 1.0:
                    tensor = tensor[tensor.shape[0] // 2:]

                trajectories[layer_key].append(to_storable_tensor(tensor))
            return hook

        # Register hooks on all target layers
        for layer_key in layer_keys:
            module = get_target_module_by_key(self.pipe.unet, layer_key)
            handles.append(module.register_forward_hook(make_hook(layer_key)))

        # Install injector if provided
        if injector is not None:
            injector.install(self.pipe)

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)

            def step_callback(pipe, step_idx, timestep, callback_kwargs):
                hook_step["i"] = step_idx
                return callback_kwargs

            result = self.pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
                callback_on_step_end=step_callback,
            )
            img = result.images[0]
        finally:
            for handle in handles:
                handle.remove()
            if injector is not None:
                injector.uninstall()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Process trajectories into MultiLayerTraceResult
        non_empty = [len(v) for v in trajectories.values() if len(v) > 0]
        if not non_empty:
            raise RuntimeError(f"No activations captured during generation for: {prompt}")

        min_steps = min(non_empty)
        used_step_indices = list(range(cfg.discovery_start_step, cfg.discovery_start_step + min_steps))

        layer_traces = {}
        for layer_key, steps_data in trajectories.items():
            layer_traces[layer_key] = [normalize_vec(step_tensor[0].float()) for step_tensor in steps_data[:min_steps]]

        trace = MultiLayerTraceResult(
            prompt=prompt,
            seed=seed,
            layer_traces=layer_traces,
            used_step_indices=used_step_indices,
        )

        return img, trace


def capture_robust_trace(
    manager: SDXLPipelineManager,
    prompt: str,
    seeds: List[int],
    cfg: CaptureConfig
) -> TraceResult:
    """
    Executes a high-throughput screen across multiple seeds to marginalize
    out the random noise distribution, returning a noise-free mean trace.
    """
    all_step_vectors = []
    used_indices = None

    for s in seeds:
        trace = manager.capture_trace(prompt, s, cfg)
        # Stack the steps for this seed into a tensor of shape [num_steps, dimensions]
        all_step_vectors.append(torch.stack(trace.per_step_vectors, dim=0))
        if used_indices is None:
            used_indices = trace.used_step_indices

    # Stack all seeds: shape becomes [num_seeds, num_steps, dimensions]
    stacked_traces = torch.stack(all_step_vectors, dim=0)

    # Marginalize out the seed noise by calculating the mean across dimension 0
    robust_mean_steps = stacked_traces.mean(dim=0)

    # Convert back to a list of tensors to match the TraceResult specification
    robust_step_list = [robust_mean_steps[i] for i in range(robust_mean_steps.shape[0])]

    return TraceResult(
        prompt=prompt,
        seed=seeds[0],
        per_step_vectors=robust_step_list,
        used_step_indices=used_indices
    )


# =========================================================
# Multi-Layer Capture & Heatmap Analysis
# =========================================================

def get_target_module_by_key(unet, layer_key: str):
    """
    Resolve layer key to UNet module.
    Extends identification.get_target_module with support for down_blocks.
    """
    # Handle down_blocks (not in identification.py)
    if layer_key == "down_0":
        return unet.down_blocks[0]
    if layer_key == "down_1":
        return unet.down_blocks[1]
    if layer_key == "down_2":
        return unet.down_blocks[2]
    # Delegate to identification module for standard layers
    return get_target_module(unet, layer_key)


# to_storable_tensor is imported from identification module for reproducibility


def capture_multi_layer_trace(
    manager: SDXLPipelineManager,
    prompt: str,
    seed: int,
    cfg: CaptureConfig,
) -> MultiLayerTraceResult:
    """
    Capture activation trajectories from multiple UNet layers simultaneously.
    Returns a MultiLayerTraceResult with per-layer, per-step vectors.
    """
    layer_keys = cfg.multi_layer_keys
    trajectories = {layer_key: [] for layer_key in layer_keys}
    handles = []
    hook_step = {"i": 0}

    def make_hook(layer_key):
        def hook(_, __, output):
            step_idx = hook_step["i"]

            if step_idx < cfg.discovery_start_step:
                return
            if cfg.discovery_end_step is not None and step_idx >= cfg.discovery_end_step:
                return

            if torch.is_tensor(output):
                tensor = output
            elif isinstance(output, (tuple, list)):
                tensors = [item for item in output if torch.is_tensor(item)]
                if not tensors:
                    return
                tensor = max(tensors, key=lambda t: t.numel())
            else:
                return

            # For CFG batches, keep only conditional branch
            if tensor.shape[0] >= 2 and cfg.guidance_scale > 1.0:
                tensor = tensor[tensor.shape[0] // 2:]

            trajectories[layer_key].append(to_storable_tensor(tensor))
        return hook

    # Register hooks on all target layers
    for layer_key in layer_keys:
        module = get_target_module_by_key(manager.pipe.unet, layer_key)
        handles.append(module.register_forward_hook(make_hook(layer_key)))

    try:
        generator = torch.Generator(device=manager.device).manual_seed(seed)

        # Track steps using a callback
        original_callback = None
        def step_callback(pipe, step_idx, timestep, callback_kwargs):
            hook_step["i"] = step_idx
            return callback_kwargs

        _ = manager.pipe(
            prompt=prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
            generator=generator,
            callback_on_step_end=step_callback,
        )
    finally:
        for handle in handles:
            handle.remove()

    # Determine common step count
    non_empty = [len(v) for v in trajectories.values() if len(v) > 0]
    if not non_empty:
        raise RuntimeError(f"No activations captured for prompt: {prompt}")

    min_steps = min(non_empty)
    used_step_indices = list(range(cfg.discovery_start_step, cfg.discovery_start_step + min_steps))

    # Normalize the layer traces
    layer_traces = {}
    for layer_key, steps_data in trajectories.items():
        layer_traces[layer_key] = [normalize_vec(step_tensor[0].float()) for step_tensor in steps_data[:min_steps]]

    return MultiLayerTraceResult(
        prompt=prompt,
        seed=seed,
        layer_traces=layer_traces,
        used_step_indices=used_step_indices,
    )


def capture_robust_multi_layer_trace(
    manager: SDXLPipelineManager,
    prompt: str,
    seeds: List[int],
    cfg: CaptureConfig
) -> MultiLayerTraceResult:
    """
    Capture multi-layer traces across multiple seeds and average them.
    """
    all_layer_traces = {layer_key: [] for layer_key in cfg.multi_layer_keys}
    used_indices = None

    for s in seeds:
        trace = capture_multi_layer_trace(manager, prompt, s, cfg)
        for layer_key in cfg.multi_layer_keys:
            # Stack steps for this seed: [num_steps, dimensions]
            stacked = torch.stack(trace.layer_traces[layer_key], dim=0)
            all_layer_traces[layer_key].append(stacked)
        if used_indices is None:
            used_indices = trace.used_step_indices

    # Average across seeds for each layer
    robust_layer_traces = {}
    for layer_key in cfg.multi_layer_keys:
        # Stack all seeds: [num_seeds, num_steps, dimensions]
        stacked_traces = torch.stack(all_layer_traces[layer_key], dim=0)
        # Marginalize out seed noise
        robust_mean_steps = stacked_traces.mean(dim=0)
        # Convert back to list of tensors
        robust_layer_traces[layer_key] = [robust_mean_steps[i] for i in range(robust_mean_steps.shape[0])]

    return MultiLayerTraceResult(
        prompt=prompt,
        seed=seeds[0],
        layer_traces=robust_layer_traces,
        used_step_indices=used_indices,
    )


def capture_multi_layer_sampled_trace(
    manager: SDXLPipelineManager,
    prompt: str,
    seeds: List[int],
    cfg: CaptureConfig
) -> MultiLayerSampledTraceResult:
    """
    Capture multi-layer traces across multiple seeds, preserving individual samples.
    Unlike capture_robust_multi_layer_trace, this does NOT average across seeds.
    Returns batched tensors of shape [num_seeds, C] per timestep.
    """
    all_layer_traces = {layer_key: [] for layer_key in cfg.multi_layer_keys}
    used_indices = None

    for s in seeds:
        trace = capture_multi_layer_trace(manager, prompt, s, cfg)
        for layer_key in cfg.multi_layer_keys:
            stacked = torch.stack(trace.layer_traces[layer_key], dim=0)  # [num_steps, C]
            all_layer_traces[layer_key].append(stacked)
        if used_indices is None:
            used_indices = trace.used_step_indices

    # Stack samples per layer: [num_seeds, num_steps, C], then split to per-step [num_seeds, C]
    layer_traces_batched = {}
    for layer_key in cfg.multi_layer_keys:
        stacked_traces = torch.stack(all_layer_traces[layer_key], dim=0)  # [num_seeds, num_steps, C]
        num_steps = stacked_traces.shape[1]
        layer_traces_batched[layer_key] = [stacked_traces[:, i, :] for i in range(num_steps)]

    return MultiLayerSampledTraceResult(
        prompt=prompt,
        seeds=seeds,
        layer_traces_batched=layer_traces_batched,
        used_step_indices=used_indices,
    )


# energy_distance_1d is imported from identification module for reproducibility


def compute_bias_for_vectors(male_vectors, female_vectors, neutral_vectors):
    """
    Compute bias metrics given batched vectors for male, female, and neutral prompts.
    Expects tensors of shape [N, C] where N is number of samples (or single vectors [C]).
    Returns continuous bias score using energy distance and steering vector.

    Wraps identification.compute_bias_for_vectors with 1D vector handling.
    """
    if male_vectors is None or female_vectors is None or neutral_vectors is None:
        return {"continuous": 0.0, "steering_vector": None}

    # Handle both single vectors and batched vectors (extend to 2D for identification module)
    if male_vectors.dim() == 1:
        male_vectors = male_vectors.unsqueeze(0)
    if female_vectors.dim() == 1:
        female_vectors = female_vectors.unsqueeze(0)
    if neutral_vectors.dim() == 1:
        neutral_vectors = neutral_vectors.unsqueeze(0)

    # Use the identification module's implementation for reproducibility
    result = _compute_bias_for_vectors_base(male_vectors, female_vectors, neutral_vectors)

    return {
        "continuous": result["continuous"],
        "steering_vector": result["steering_vector"],
    }


def compute_multi_layer_bias_analysis(
    male_trace: MultiLayerSampledTraceResult,
    female_trace: MultiLayerSampledTraceResult,
    neutral_trace: MultiLayerSampledTraceResult,
    layer_keys: List[str],
) -> List[Dict]:
    """
    Compute bias metrics per layer and per timestep using energy distance.
    Expects sampled traces with batched vectors [num_seeds, C] per timestep.
    Returns list of dicts with layer, step, continuous bias, and steering vector.
    """
    results = []

    for layer_key in layer_keys:
        male_steps = male_trace.layer_traces_batched.get(layer_key, [])
        female_steps = female_trace.layer_traces_batched.get(layer_key, [])
        neutral_steps = neutral_trace.layer_traces_batched.get(layer_key, [])

        common_steps = min(len(male_steps), len(female_steps), len(neutral_steps))

        for step in range(common_steps):
            male_vecs = male_steps[step]      # [num_seeds, C]
            female_vecs = female_steps[step]  # [num_seeds, C]
            neutral_vecs = neutral_steps[step]  # [num_seeds, C]

            bias_metrics = compute_bias_for_vectors(male_vecs, female_vecs, neutral_vecs)

            results.append({
                "layer": layer_key,
                "step": step,
                "continuous": bias_metrics["continuous"],
                "steering_vector": bias_metrics["steering_vector"],
            })

    return results


def plot_bias_heatmap(profession: str, analysis_results: List[Dict], layer_keys: List[str], output_path: str):
    """
    Create a heatmap visualization of bias across layers and timesteps.
    Similar to the identification.py heatmaps.
    """
    steps = sorted(list(set(r['step'] for r in analysis_results)))

    grid = np.zeros((len(layer_keys), len(steps)))
    layer_to_idx = {name: i for i, name in enumerate(layer_keys)}

    for res in analysis_results:
        l_idx = layer_to_idx.get(res['layer'])
        s_idx = res['step']
        if l_idx is not None and s_idx < len(steps):
            grid[l_idx, s_idx] = res['continuous']

    plt.figure(figsize=(12, 6))

    max_val = np.nanmax(np.abs(grid))
    if max_val == 0 or np.isnan(max_val):
        max_val = 1

    plt.imshow(grid, cmap='coolwarm', origin='lower', aspect='auto', vmin=-max_val, vmax=max_val)

    cbar = plt.colorbar()
    cbar.set_label('Bias (Negative=Female, Positive=Male)', rotation=270, labelpad=15)

    plt.yticks(range(len(layer_keys)), layer_keys)
    if len(steps) > 10:
        tick_indices = list(range(0, len(steps), 5))
        plt.xticks(tick_indices, [steps[i] for i in tick_indices])
    else:
        plt.xticks(range(len(steps)), steps)

    plt.xlabel('Timestep (Generation Process)')
    plt.ylabel('Network Layer')
    plt.title(f'Gender Bias Heatmap: {profession}')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    log(f"  Saved heatmap to {output_path}")


def compute_single_image_bias(
    image_trace: MultiLayerTraceResult,
    male_trace: MultiLayerSampledTraceResult,
    female_trace: MultiLayerSampledTraceResult,
    layer_keys: List[str],
) -> List[Dict]:
    """
    Compute bias for a single image's activations relative to male/female reference distributions.
    Uses energy distance between the single image point and the reference distributions.
    """
    results = []

    for layer_key in layer_keys:
        image_steps = image_trace.layer_traces.get(layer_key, [])
        male_steps = male_trace.layer_traces_batched.get(layer_key, [])
        female_steps = female_trace.layer_traces_batched.get(layer_key, [])

        common_steps = min(len(image_steps), len(male_steps), len(female_steps))

        for step in range(common_steps):
            image_vec = image_steps[step]      # [C]
            male_vecs = male_steps[step]       # [num_seeds, C]
            female_vecs = female_steps[step]   # [num_seeds, C]

            # Normalize
            image_norm = torch.nn.functional.normalize(image_vec.unsqueeze(0), p=2, dim=1)
            male_norm = torch.nn.functional.normalize(male_vecs, p=2, dim=1)
            female_norm = torch.nn.functional.normalize(female_vecs, p=2, dim=1)

            # Compute axis from reference distributions
            male_center = torch.nn.functional.normalize(male_norm.mean(dim=0, keepdim=True), p=2, dim=1)
            female_center = torch.nn.functional.normalize(female_norm.mean(dim=0, keepdim=True), p=2, dim=1)
            axis = torch.nn.functional.normalize(male_center - female_center, p=2, dim=1)

            # Project onto axis
            image_score = [(image_norm * axis).sum().item()]
            male_scores = (male_norm * axis).sum(dim=1).tolist()
            female_scores = (female_norm * axis).sum(dim=1).tolist()

            # Compute energy distance from single point to distributions
            d_im = energy_distance_1d(image_score, male_scores)
            d_if = energy_distance_1d(image_score, female_scores)
            continuous = (d_if - d_im) / (d_if + d_im + 1e-8)

            results.append({
                "layer": layer_key,
                "step": step,
                "continuous": continuous,
            })

    return results


def plot_steered_comparison_heatmap(
    profession: str,
    baseline_results: List[Dict],
    masculine_results: List[Dict],
    feminine_results: List[Dict],
    layer_keys: List[str],
    output_path: str,
    strength: float,
):
    """
    Create a side-by-side comparison heatmap showing baseline vs steered images.
    """
    steps = sorted(list(set(r['step'] for r in baseline_results)))
    n_layers = len(layer_keys)
    n_steps = len(steps)

    # Create grids for each condition
    baseline_grid = np.zeros((n_layers, n_steps))
    masculine_grid = np.zeros((n_layers, n_steps))
    feminine_grid = np.zeros((n_layers, n_steps))

    layer_to_idx = {name: i for i, name in enumerate(layer_keys)}

    for res in baseline_results:
        l_idx = layer_to_idx.get(res['layer'])
        s_idx = res['step']
        if l_idx is not None and s_idx < n_steps:
            baseline_grid[l_idx, s_idx] = res['continuous']

    for res in masculine_results:
        l_idx = layer_to_idx.get(res['layer'])
        s_idx = res['step']
        if l_idx is not None and s_idx < n_steps:
            masculine_grid[l_idx, s_idx] = res['continuous']

    for res in feminine_results:
        l_idx = layer_to_idx.get(res['layer'])
        s_idx = res['step']
        if l_idx is not None and s_idx < n_steps:
            feminine_grid[l_idx, s_idx] = res['continuous']

    # Find global min/max for consistent colorscale
    all_vals = np.concatenate([baseline_grid.flatten(), masculine_grid.flatten(), feminine_grid.flatten()])
    max_val = np.nanmax(np.abs(all_vals))
    if max_val == 0 or np.isnan(max_val):
        max_val = 1

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    titles = ['Baseline (No Steering)', f'Masculine Steered (s={strength:.2f})', f'Feminine Steered (s={strength:.2f})']
    grids = [baseline_grid, masculine_grid, feminine_grid]

    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, cmap='coolwarm', origin='lower', aspect='auto', vmin=-max_val, vmax=max_val)
        ax.set_yticks(range(n_layers))
        ax.set_yticklabels(layer_keys)
        if n_steps > 10:
            tick_indices = list(range(0, n_steps, 5))
            ax.set_xticks(tick_indices)
            ax.set_xticklabels([steps[i] for i in tick_indices])
        else:
            ax.set_xticks(range(n_steps))
            ax.set_xticklabels(steps)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Layer')
        ax.set_title(title)

    fig.suptitle(f'Gender Bias Comparison: {profession}', fontsize=14)
    fig.colorbar(im, ax=axes, label='Bias (Negative=Female, Positive=Male)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    log(f"  Saved comparison heatmap to {output_path}")


# =========================================================
# Progression Plot (integrated from plot_injection.py)
# =========================================================

@dataclass
class ImageItem:
    path: Path
    axis: str
    power: Optional[float]
    power_text: str
    seed: Optional[int] = None
    is_baseline: bool = False


def parse_power_token(token: str) -> Tuple[float, str]:
    """Parse power token like 's10p0' into float 10.0 and display text '10.0'."""
    cleaned = token.strip()
    if cleaned.startswith("s"):
        cleaned = cleaned[1:]
    power_text = cleaned.replace("p", ".")
    return float(power_text), power_text


def parse_seed_token(token: str) -> Optional[int]:
    """Parse seed token like 'seed0' into int 0."""
    if token.startswith("seed"):
        return int(token[4:])
    return None


def parse_image_name(image_path: Path) -> ImageItem:
    """Parse image filename into ImageItem with axis, power, seed info."""
    stem = image_path.stem
    parts = stem.split("_")

    # Check for baseline
    if "baseline" in stem:
        seed = None
        for part in parts:
            if part.startswith("seed"):
                seed = parse_seed_token(part)
        return ImageItem(
            path=image_path,
            axis="baseline",
            power=None,
            power_text="0.0",
            seed=seed,
            is_baseline=True,
        )

    # Parse axis (masculine/feminine)
    axis = parts[0]
    power_value = None
    power_text = ""
    seed = None

    for part in parts[1:]:
        if part.startswith("s") and not part.startswith("seed"):
            power_value, power_text = parse_power_token(part)
        elif part.startswith("seed"):
            seed = parse_seed_token(part)

    return ImageItem(
        path=image_path,
        axis=axis,
        power=power_value,
        power_text=power_text,
        seed=seed,
        is_baseline=False,
    )


def build_progression_sequence(
    by_axis: Dict[str, List[ImageItem]],
    baseline: Optional[ImageItem],
) -> Tuple[List[ImageItem], str]:
    """Build progression row: feminine (descending) -> baseline -> masculine (ascending)"""
    sequence = []

    # Feminine side (descending strength = strongest first)
    if "feminine" in by_axis:
        feminine = sorted(
            by_axis["feminine"],
            key=lambda x: (x.power is None, x.power),
            reverse=True,
        )
        sequence.extend(feminine)

    # Baseline in middle
    if baseline is not None:
        sequence.append(baseline)

    # Masculine side (ascending strength = weakest first)
    if "masculine" in by_axis:
        masculine = sorted(
            by_axis["masculine"],
            key=lambda x: (x.power is None, x.power),
        )
        sequence.extend(masculine)

    subtitle = "Feminine ← Baseline → Masculine"
    return sequence, subtitle


def panel_label_for_progression(item: ImageItem) -> str:
    """Generate label for image panel."""
    if item.is_baseline:
        return "baseline"
    return f"{item.axis}\ns={item.power_text}"


def plot_profession_progression(
    prof_dir: Path,
    output_dir: Path,
    seed: Optional[int] = None,
) -> Optional[Path]:
    """Create a progression plot for a single profession with images and heatmaps."""
    from PIL import Image

    image_paths = sorted(prof_dir.glob("*.png"))
    if not image_paths:
        return None

    parsed = [parse_image_name(path) for path in image_paths]

    # Filter by seed if specified
    if seed is not None:
        parsed = [item for item in parsed if item.seed == seed]

    if not parsed:
        return None

    baseline = next((item for item in parsed if item.is_baseline), None)

    by_axis: Dict[str, List[ImageItem]] = {}
    for item in parsed:
        if item.is_baseline:
            continue
        by_axis.setdefault(item.axis, []).append(item)

    if not by_axis and baseline is None:
        return None

    sequence, subtitle = build_progression_sequence(by_axis, baseline)

    if not sequence:
        return None

    # Find heatmaps directory (sibling to generations directory)
    heatmaps_dir = prof_dir.parent / "bias_heatmaps"
    prof_slug = prof_dir.name.replace(" ", "_")
    seed_suffix = f"_seed{seed}" if seed is not None else ""

    cols = len(sequence)
    # Create 2 rows: top for images, bottom for heatmaps
    fig, axes = plt.subplots(2, cols, figsize=(3 * cols, 7.5), squeeze=False)

    for idx, item in enumerate(sequence):
        # Top row: images
        ax_img = axes[0][idx]
        with Image.open(item.path) as img:
            ax_img.imshow(img)
        ax_img.axis("off")
        ax_img.set_title(panel_label_for_progression(item), fontsize=10)

        # Bottom row: heatmaps
        ax_heatmap = axes[1][idx]

        # Construct heatmap filename based on item type
        if item.is_baseline:
            heatmap_filename = f"{prof_slug}_baseline{seed_suffix}.png"
        else:
            heatmap_filename = f"{prof_slug}_{item.axis}_s{item.power_text}{seed_suffix}.png"

        heatmap_path = heatmaps_dir / heatmap_filename

        print(f"Looking for heatmap at: {heatmap_path}")

        # Display heatmap if it exists, otherwise show a placeholder
        if heatmap_path.exists():
            with Image.open(heatmap_path) as heatmap_img:
                ax_heatmap.imshow(heatmap_img)
        else:
            # Show a blank placeholder if heatmap doesn't exist
            ax_heatmap.text(0.5, 0.5, "No heatmap", ha="center", va="center", fontsize=8)
        ax_heatmap.axis("off")

    seed_str = f" (seed={seed})" if seed is not None else ""
    fig.suptitle(f"{prof_dir.name}{seed_str}\n{subtitle}", fontsize=12, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{prof_dir.name}_progression{seed_suffix}.png"
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out_path


# =========================================================
# Axis construction (Contextual Scrubbing)
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
# Prompt helpers (matching identification.py style)
# =========================================================

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


def make_prompts(profession: str) -> Dict[str, str]:
    return {
        "neutral": NEUTRAL_PROFESSION_PROMPT_TEMPLATE.format(profession=profession),
        "male": MALE_PROFESSION_PROMPT_TEMPLATE.format(profession=profession),
        "female": FEMALE_PROFESSION_PROMPT_TEMPLATE.format(profession=profession),
    }


# =========================================================
# CLI
# =========================================================

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

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--enable-cpu-offload", action="store_true")

    p.add_argument("--module-path", type=str, default="unet.down_blocks.2")
    p.add_argument(
        "--inject-layer",
        type=str,
        default="down_2",
        choices=["mid_block", "down_0", "down_1", "down_2", "up_0", "up_1", "up_2"],
    )

    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--mask-sigma", type=float, default=0.4)

    p.add_argument("--discovery-start-step", type=int, default=0)
    p.add_argument("--discovery-end-step", type=int, default=-1, help="Use -1 for no limit")
    p.add_argument("--num-discovery-seeds", type=int, default=5, help="Number of seeds to average for robust extraction")

    p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="cond")
    p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
    p.add_argument("--inject-start-step", type=int, default=0)
    p.add_argument("--inject-end-step", type=int, default=12, help="Use -1 for no limit")
    p.add_argument("--inject-schedule", type=str, choices=["flat", "linear_decay", "cosine_decay"], default="flat")
    p.add_argument("--inject-log-every", type=int, default=0)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--professions", type=str, nargs="+", default=PROFESSIONS)
    p.add_argument("--min-strength", type=float, default=0.5)
    p.add_argument("--max-strength", type=float, default=2.0)
    p.add_argument("--num-strengths", type=int, default=4)
    p.add_argument("--strengths", type=float, nargs="+", default=[10, 30, 50], help="Directly specify strength values to use (overrides min/max/num)")

    p.add_argument("--output-dir", type=str, default="outputs/robust_independent_injection")
    p.add_argument("--multi-layer-keys", type=str, nargs="+", default=["mid_block", "up_0", "up_1", "up_2"],
                   help="Layers to extract for heatmap analysis")
    p.add_argument("--skip-heatmaps", action="store_true", help="Skip multi-layer heatmap generation")
    p.add_argument("--skip-progression-plots", action="store_true", help="Skip progression plot generation")

    return p.parse_args()


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)

    capture_cfg = CaptureConfig(
        module_path=args.module_path,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        mask_sigma=args.mask_sigma,
        discovery_start_step=args.discovery_start_step,
        discovery_end_step=None if args.discovery_end_step < 0 else args.discovery_end_step,
        save_all_timestep_vectors=True,
        multi_layer_keys=args.multi_layer_keys,
    )

    # Create heatmaps directory
    heatmaps_dir = Path(args.output_dir) / "bias_heatmaps"
    ensure_dir(heatmaps_dir)

    manager = SDXLPipelineManager(
        model_id=args.model_id,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    global_meta = {
        "seed": args.seed,
        "model_id": args.model_id,
        "num_discovery_seeds": args.num_discovery_seeds,
        "capture_cfg": asdict(capture_cfg),
        "inject_layer": args.inject_layer,
        "inject_cfg_target": args.inject_cfg_target,
        "inject_normalize": args.inject_normalize,
        "inject_start_step": args.inject_start_step,
        "inject_end_step": None if args.inject_end_step < 0 else args.inject_end_step,
        "inject_schedule": args.inject_schedule,
        "min_strength": args.min_strength,
        "max_strength": args.max_strength,
        "num_strengths": args.num_strengths,
        "strengths": args.strengths,
        "professions": args.professions,
        "multi_layer_keys": args.multi_layer_keys,
    }
    save_json(Path(args.output_dir) / "run_config.json", global_meta)

    log(f"[start] Beginning robust high-throughput discovery (N={args.num_discovery_seeds} seeds)")

    discovery_seeds = [args.seed + i for i in range(args.num_discovery_seeds)]

    # Create progression_dir early so we can generate plots after each profession
    progression_dir = Path(args.output_dir) / "progression_plots"
    if not args.skip_progression_plots:
        ensure_dir(progression_dir)

    for profession in args.professions:
        prof_slug = slugify(profession)
        prof_dir = Path(args.output_dir) / prof_slug
        ensure_dir(prof_dir)

        prompts = make_prompts(profession)

        log(f"\n[capture] Profession: {profession}")
        log(f"  neutral: {prompts['neutral']}")
        log(f"  male:    {prompts['male']}")
        log(f"  female:  {prompts['female']}")

        neutral_trace = capture_robust_trace(manager, prompts["neutral"], discovery_seeds, capture_cfg)
        male_trace = capture_robust_trace(manager, prompts["male"], discovery_seeds, capture_cfg)
        female_trace = capture_robust_trace(manager, prompts["female"], discovery_seeds, capture_cfg)

        neutral_mean = mean_trace_vector(neutral_trace)
        male_mean = mean_trace_vector(male_trace)
        female_mean = mean_trace_vector(female_trace)
        
        vec_masculine, vec_feminine = build_independent_vectors(neutral_trace, male_trace, female_trace)

        # Save trace vectors
        torch_save(prof_dir / "neutral_mean.pt", neutral_mean)
        torch_save(prof_dir / "male_mean.pt", male_mean)
        torch_save(prof_dir / "female_mean.pt", female_mean)
        torch_save(prof_dir / "vec_masculine.pt", vec_masculine)
        torch_save(prof_dir / "vec_feminine.pt", vec_feminine)

        if capture_cfg.save_all_timestep_vectors:
            torch_save(prof_dir / "neutral_steps.pt", torch.stack(neutral_trace.per_step_vectors))
            torch_save(prof_dir / "male_steps.pt", torch.stack(male_trace.per_step_vectors))
            torch_save(prof_dir / "female_steps.pt", torch.stack(female_trace.per_step_vectors))

        save_json(
            prof_dir / "trace_metadata.json",
            {
                "profession": profession,
                "prompts": prompts,
                "discovery_seeds": discovery_seeds,
                "neutral_used_steps": neutral_trace.used_step_indices,
                "male_used_steps": male_trace.used_step_indices,
                "female_used_steps": female_trace.used_step_indices,
            },
        )

        # =========================================================
        # Multi-Layer Heatmap Analysis
        # =========================================================
        if not args.skip_heatmaps:
            log(f"[multi-layer] Capturing activations from layers: {args.multi_layer_keys}")

            neutral_multi_trace = capture_multi_layer_sampled_trace(
                manager, prompts["neutral"], discovery_seeds, capture_cfg
            )
            male_multi_trace = capture_multi_layer_sampled_trace(
                manager, prompts["male"], discovery_seeds, capture_cfg
            )
            female_multi_trace = capture_multi_layer_sampled_trace(
                manager, prompts["female"], discovery_seeds, capture_cfg
            )

            # Compute bias analysis across all layers and timesteps
            analysis_results = compute_multi_layer_bias_analysis(
                male_multi_trace, female_multi_trace, neutral_multi_trace, args.multi_layer_keys
            )

            # Save multi-layer traces (stacked across seeds: [num_seeds, num_steps, C])
            multi_layer_dir = prof_dir / "multi_layer_traces"
            ensure_dir(multi_layer_dir)

            for layer_key in args.multi_layer_keys:
                torch_save(
                    multi_layer_dir / f"neutral_{layer_key}_steps.pt",
                    torch.stack(neutral_multi_trace.layer_traces_batched[layer_key])
                )
                torch_save(
                    multi_layer_dir / f"male_{layer_key}_steps.pt",
                    torch.stack(male_multi_trace.layer_traces_batched[layer_key])
                )
                torch_save(
                    multi_layer_dir / f"female_{layer_key}_steps.pt",
                    torch.stack(female_multi_trace.layer_traces_batched[layer_key])
                )

            # Save analysis results (without steering vectors for JSON)
            analysis_for_json = [
                {"layer": r["layer"], "step": r["step"], "continuous": r["continuous"]}
                for r in analysis_results
            ]
            save_json(prof_dir / "multi_layer_bias_analysis.json", analysis_for_json)

            # Plot heatmap
            heatmap_path = heatmaps_dir / f"{prof_slug}.png"
            plot_bias_heatmap(profession, analysis_results, args.multi_layer_keys, str(heatmap_path))

            # Print summary statistics
            avg_bias = sum(r['continuous'] for r in analysis_results) / len(analysis_results) if analysis_results else 0
            max_bias_entry = max(analysis_results, key=lambda x: abs(x['continuous'])) if analysis_results else None
            log(f"  [MULTI-LAYER BIAS SUMMARY] Avg: {avg_bias:.4f}")
            if max_bias_entry:
                log(f"  Max Bias: {max_bias_entry['continuous']:.4f} at {max_bias_entry['layer']} step {max_bias_entry['step']}")

        # =========================================================
        # Baseline & Steered Image Generation with Multi-Layer Capture
        # Generate for ALL discovery seeds (not just one)
        # =========================================================

        if args.strengths is not None and len(args.strengths) > 0:
            strength_values = args.strengths
        else:
            strength_values = np.linspace(args.min_strength, args.max_strength, args.num_strengths)

        for gen_seed in discovery_seeds:
            log(f"[generate] Baseline for {profession} (seed={gen_seed})")
            seed_suffix = f"_seed{gen_seed}"

            if not args.skip_heatmaps:
                # Generate baseline WITH multi-layer capture
                baseline, baseline_trace = manager.generate_with_multi_layer_capture(
                    prompts["neutral"], gen_seed, capture_cfg, injector=None
                )
                baseline.save(prof_dir / f"baseline_neutral{seed_suffix}.png")

                # Compute baseline bias using male/female references
                baseline_bias = compute_single_image_bias(
                    baseline_trace, male_multi_trace, female_multi_trace, args.multi_layer_keys
                )
                save_json(prof_dir / f"baseline_bias_analysis{seed_suffix}.json", baseline_bias)

                # Save baseline heatmap
                baseline_heatmap_path = heatmaps_dir / f"{prof_slug}_baseline{seed_suffix}.png"
                plot_bias_heatmap(f"{profession} (Baseline seed={gen_seed})", baseline_bias, args.multi_layer_keys, str(baseline_heatmap_path))
            else:
                baseline = manager.generate(prompts["neutral"], gen_seed, capture_cfg)
                baseline.save(prof_dir / f"baseline_neutral{seed_suffix}.png")

            for strength in strength_values:
                strength_str = str(round(strength, 2)).replace('.', 'p')

                log(f"[generate] {profession} | MASCULINE | strength={strength:.2f} | seed={gen_seed}")
                inj_cfg_pos = InjectorConfig(
                    layer_key=args.inject_layer,
                    strength=strength,
                    cfg_target=args.inject_cfg_target,
                    normalize=args.inject_normalize,
                    start_step=args.inject_start_step,
                    end_step=None if args.inject_end_step < 0 else args.inject_end_step,
                    schedule=args.inject_schedule,
                    mask_sigma=args.mask_sigma,
                    log_every=args.inject_log_every,
                )
                injector_pos = ActivationInjector(inj_cfg_pos, vec_masculine)

                log(f"[generate] {profession} | FEMININE | strength={strength:.2f} | seed={gen_seed}")
                inj_cfg_neg = InjectorConfig(
                    layer_key=args.inject_layer,
                    strength=strength,
                    cfg_target=args.inject_cfg_target,
                    normalize=args.inject_normalize,
                    start_step=args.inject_start_step,
                    end_step=None if args.inject_end_step < 0 else args.inject_end_step,
                    schedule=args.inject_schedule,
                    mask_sigma=args.mask_sigma,
                    log_every=args.inject_log_every,
                )
                injector_neg = ActivationInjector(inj_cfg_neg, vec_feminine)

                if not args.skip_heatmaps:
                    # Generate WITH multi-layer capture
                    img_pos, masculine_trace = manager.generate_with_multi_layer_capture(
                        prompts["neutral"], gen_seed, capture_cfg, injector=injector_pos
                    )
                    img_pos.save(prof_dir / f"masculine_s{strength_str}{seed_suffix}.png")

                    img_neg, feminine_trace = manager.generate_with_multi_layer_capture(
                        prompts["neutral"], gen_seed, capture_cfg, injector=injector_neg
                    )
                    img_neg.save(prof_dir / f"feminine_s{strength_str}{seed_suffix}.png")

                    # Compute bias for steered images
                    masculine_bias = compute_single_image_bias(
                        masculine_trace, male_multi_trace, female_multi_trace, args.multi_layer_keys
                    )
                    feminine_bias = compute_single_image_bias(
                        feminine_trace, male_multi_trace, female_multi_trace, args.multi_layer_keys
                    )

                    # Save individual steered heatmaps
                    masculine_heatmap_path = heatmaps_dir / f"{prof_slug}_masculine_s{strength_str}{seed_suffix}.png"
                    plot_bias_heatmap(
                        f"{profession} (Masculine s={strength:.2f} seed={gen_seed})",
                        masculine_bias, args.multi_layer_keys, str(masculine_heatmap_path)
                    )

                    feminine_heatmap_path = heatmaps_dir / f"{prof_slug}_feminine_s{strength_str}{seed_suffix}.png"
                    plot_bias_heatmap(
                        f"{profession} (Feminine s={strength:.2f} seed={gen_seed})",
                        feminine_bias, args.multi_layer_keys, str(feminine_heatmap_path)
                    )

                    # Save comparison heatmap (baseline vs masculine vs feminine)
                    comparison_heatmap_path = heatmaps_dir / f"{prof_slug}_comparison_s{strength_str}{seed_suffix}.png"
                    plot_steered_comparison_heatmap(
                        profession,
                        baseline_bias,
                        masculine_bias,
                        feminine_bias,
                        args.multi_layer_keys,
                        str(comparison_heatmap_path),
                        strength,
                    )

                    # Save analysis JSON
                    save_json(prof_dir / f"masculine_s{strength_str}{seed_suffix}_bias_analysis.json", masculine_bias)
                    save_json(prof_dir / f"feminine_s{strength_str}{seed_suffix}_bias_analysis.json", feminine_bias)

                    # Print summary
                    baseline_avg = sum(r['continuous'] for r in baseline_bias) / len(baseline_bias) if baseline_bias else 0
                    masculine_avg = sum(r['continuous'] for r in masculine_bias) / len(masculine_bias) if masculine_bias else 0
                    feminine_avg = sum(r['continuous'] for r in feminine_bias) / len(feminine_bias) if feminine_bias else 0
                    log(f"  [BIAS SHIFT] Baseline: {baseline_avg:.4f} -> Masculine: {masculine_avg:.4f}, Feminine: {feminine_avg:.4f}")

                    steering_worked = (masculine_avg > 0) if baseline_avg <= 0 else (feminine_avg < 0)
                    log(f"  [STEERING EFFECT] {'SUCCESS' if steering_worked else 'FAILURE'}")
                else:
                    img_pos = manager.generate(prompts["neutral"], gen_seed, capture_cfg, injector=injector_pos)
                    img_pos.save(prof_dir / f"masculine_s{strength_str}{seed_suffix}.png")

                    img_neg = manager.generate(prompts["neutral"], gen_seed, capture_cfg, injector=injector_neg)
                    img_neg.save(prof_dir / f"feminine_s{strength_str}{seed_suffix}.png")

        # Generate progression plots for this profession immediately
        if not args.skip_progression_plots:
            for gen_seed in discovery_seeds:
                out = plot_profession_progression(prof_dir, progression_dir, seed=gen_seed)
                if out is not None:
                    log(f"  [plot] Saved progression: {out.name}")

    log("[done] All independent injections complete.")


if __name__ == "__main__":
    main()