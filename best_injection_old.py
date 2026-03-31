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
from diffusers import StableDiffusionXLPipeline


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
# Prompt helpers
# =========================================================

def make_prompts(profession: str) -> Dict[str, str]:
    common = "studio lighting, face clearly visible, realistic profile picture, centered subject"
    return {
        "neutral": f"A {common} image of a {profession}",
        "male": f"A {common} image of a male {profession}",
        "female": f"A {common} image of a female {profession}",
    }


# =========================================================
# CLI
# =========================================================

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
    p.add_argument("--discovery-end-step", type=int, default=12, help="Use -1 for no limit")
    p.add_argument("--num-discovery-seeds", type=int, default=5, help="Number of seeds to average for robust extraction")

    p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="cond")
    p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
    p.add_argument("--inject-start-step", type=int, default=0)
    p.add_argument("--inject-end-step", type=int, default=12, help="Use -1 for no limit")
    p.add_argument("--inject-schedule", type=str, choices=["flat", "linear_decay", "cosine_decay"], default="flat")
    p.add_argument("--inject-log-every", type=int, default=0)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--professions", type=str, nargs="+", default=["Nurse", "Doctor", "Programmer", "Kindergarten Teacher", "CEO", "Construction Worker", "Flight Attendant", "Scientist", "Firefighter", "Librarian"])
    p.add_argument("--min-strength", type=float, default=0.5)
    p.add_argument("--max-strength", type=float, default=2.0)
    p.add_argument("--num-strengths", type=int, default=4)

    p.add_argument("--output-dir", type=str, default="outputs/robust_independent_injection")

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
    )

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
        "professions": args.professions,
    }
    save_json(Path(args.output_dir) / "run_config.json", global_meta)

    log(f"[start] Beginning robust high-throughput discovery (N={args.num_discovery_seeds} seeds)")
    
    discovery_seeds = [args.seed + i for i in range(args.num_discovery_seeds)]

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

        # Baseline
        log(f"[generate] Baseline for {profession} (seed={args.seed})")
        baseline = manager.generate(prompts["neutral"], args.seed, capture_cfg)
        baseline.save(prof_dir / "baseline_neutral.png")

        for strength in np.linspace(args.min_strength, args.max_strength, args.num_strengths):
            log(f"[generate] {profession} | MASCULINE | strength={strength:.2f}")
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
            img_pos = manager.generate(prompts["neutral"], args.seed, capture_cfg, injector=injector_pos)
            img_pos.save(prof_dir / f"masculine_s{str(round(strength, 2)).replace('.', 'p')}.png")

            log(f"[generate] {profession} | FEMININE | strength={strength:.2f}")
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
            img_neg = manager.generate(prompts["neutral"], args.seed, capture_cfg, injector=injector_neg)
            img_neg.save(prof_dir / f"feminine_s{str(round(strength, 2)).replace('.', 'p')}.png")

    log("[done] All independent injections complete.")


if __name__ == "__main__":
    main()