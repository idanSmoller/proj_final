#!/usr/bin/env python3
"""
Gender Bias Vector Injection (Merged & Optimized - Memory Safe)

This script combines the robust spatial broadcasting of older experiments with 
the flexible hook architecture and timestep scheduling of the newer iterations.

Features:
- 'broadcast' vs 'projected' injection styles.
- Strict VRAM management (VAE slicing, GC) to prevent SDXL OOMs.
- Fast Mean Delta discovery.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from diffusers import StableDiffusionXLPipeline
from tqdm.auto import tqdm


# =========================================================
# Logging / utilities
# =========================================================

def log(msg: str) -> None:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {msg}", flush=True)


def set_global_seed(seed: int) -> None:
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_default_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        return torch.float16
    return torch.float32


def normalize_vec(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    norm = x.norm(p=2)
    if norm.item() == 0:
        return x
    return x / (norm + 1e-8)


def slugify(text: str, max_len: int = 80) -> str:
    text = text.strip().lower().replace(" ", "_")
    safe = "".join(ch for ch in text if ch.isalnum() or ch in "_-")
    return safe[:max_len]


# =========================================================
# Configs and results
# =========================================================

@dataclass
class CaptureConfig:
    module_path: str = "unet.mid_block"
    num_inference_steps: int = 30
    guidance_scale: float = 5.0
    height: int = 1024
    width: int = 1024
    use_mean_std_pool: bool = True
    debug: bool = False
    hook_log_every: int = 5
    show_diffusers_progress: bool = True


@dataclass
class InjectorConfig:
    layer_key: str = "mid_block"
    injection_style: str = "broadcast"    # "broadcast" or "projected"
    mode: str = "add"                     # "add", "replace", "blend"
    strength: float = 1.0
    cfg_target: str = "both"              # "cond", "uncond", "both"
    normalize: str = "rms"                # "none", "rms"
    start_step: int = 0
    end_step: Optional[int] = None
    schedule: str = "flat"                # "flat", "linear", "cosine", "sine"
    blend_alpha: float = 0.5
    mean_std_strategy: str = "truncate"   # "truncate" or "mean_plus_std"
    target_delta_ratio: float = 0.0       # 0 disables autoscaling (Recommended for identity shifts)
    auto_scale_mode: str = "cap"          
    max_auto_scale: float = 80.0
    log_every: int = 5


@dataclass
class TraceResult:
    prompt: str
    seed: int
    timestep_vectors: List[torch.Tensor]


@dataclass
class ProbeFitResult:
    weight: torch.Tensor
    bias: float
    train_accuracy: float


@dataclass
class VectorDiscoveryResult:
    probe_weight: torch.Tensor
    probe_bias: float
    probe_train_accuracy: float
    mean_delta_vector: torch.Tensor


# =========================================================
# Activation Injector
# =========================================================

class ActivationInjector:
    def __init__(self, cfg: InjectorConfig, activation: torch.Tensor):
        self.cfg = cfg
        self.activation = activation.detach().float().cpu()
        self.current_step = 0
        self.injected_steps = 0
        self.handle = None
        self.target_module = None

    def install(self, pipe: StableDiffusionXLPipeline) -> "ActivationInjector":
        self.target_module = self._resolve_target_module(pipe)
        self.handle = self.target_module.register_forward_hook(self._hook_fn)
        self.current_step = 0
        self.injected_steps = 0
        log(
            f"[injector] Installed on {self.cfg.layer_key}, style={self.cfg.injection_style}, "
            f"strength={self.cfg.strength}, cfg_target={self.cfg.cfg_target}"
        )
        return self

    def uninstall(self) -> None:
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            log(
                f"[injector] Uninstalled after {self.current_step} hook calls, "
                f"injected on {self.injected_steps} steps"
            )

    def _resolve_target_module(self, pipe: StableDiffusionXLPipeline) -> torch.nn.Module:
        unet = pipe.unet
        layer_key = self.cfg.layer_key

        if layer_key == "mid_block":
            return unet.mid_block
        if layer_key.startswith("up_"):
            idx = int(layer_key.split("_")[1])
            return unet.up_blocks[idx]
        if layer_key.startswith("down_"):
            idx = int(layer_key.split("_")[1])
            return unet.down_blocks[idx]

        raise ValueError(f"Unknown layer_key: {layer_key}")

    def _hook_fn(self, module, inputs, output):
        if not self._should_inject():
            self.current_step += 1
            return output

        tensor, where = self._extract_main_tensor(output)
        if tensor is None or tensor.dim() != 4:
            self.current_step += 1
            return output

        modified = self._route_injection(tensor)
        self.injected_steps += 1
        self._maybe_log_effect(tensor, modified)
        self.current_step += 1
        return self._reinsert_tensor(output, where, modified)

    def _extract_main_tensor(self, output):
        if torch.is_tensor(output): return output, None
        if isinstance(output, tuple):
            for i, item in enumerate(output):
                if torch.is_tensor(item): return item, ("tuple", i)
        if isinstance(output, list):
            for i, item in enumerate(output):
                if torch.is_tensor(item): return item, ("list", i)
        return None, None

    def _reinsert_tensor(self, output, where, new_tensor):
        if where is None: return new_tensor
        kind, idx = where
        if kind == "tuple":
            out = list(output)
            out[idx] = new_tensor
            return tuple(out)
        if kind == "list":
            output[idx] = new_tensor
            return output
        return output

    def _should_inject(self) -> bool:
        if self.current_step < self.cfg.start_step: return False
        if self.cfg.end_step is not None and self.current_step >= self.cfg.end_step: return False
        return True

    def _step_scale(self) -> float:
        if self.cfg.end_step is None: return 1.0
        length = max(self.cfg.end_step - self.cfg.start_step, 1)
        pos = self.current_step - self.cfg.start_step
        if length <= 1:
            t = 1.0
        else:
            t = max(0.0, min(1.0, pos / (length - 1)))

        if self.cfg.schedule == "flat": return 1.0
        if self.cfg.schedule == "linear": return 1.0 - t
        if self.cfg.schedule == "cosine": return 0.5 * (1.0 - math.cos(t * math.pi))
        if self.cfg.schedule == "sine": return math.sin(t * math.pi)
        return 1.0

    def _prepare_direction(self, expected_channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        v = self.activation.to(device, dtype=dtype).flatten()
        
        if v.numel() == expected_channels * 2:
            if self.cfg.mean_std_strategy == "truncate":
                v = v[:expected_channels]
            elif self.cfg.mean_std_strategy == "mean_plus_std":
                v = v[:expected_channels] + 0.25 * v[expected_channels:]
        
        if v.numel() != expected_channels:
            raise ValueError(f"Channel mismatch: steering has {v.numel()} values, tensor has {expected_channels}")

        return normalize_vec(v).to(device=device, dtype=dtype)

    def _route_injection(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.cfg.cfg_target == "both" or tensor.shape[0] < 2 or tensor.shape[0] % 2 != 0:
            return self._apply_injection(tensor)

        half = tensor.shape[0] // 2
        out = tensor.clone()

        if self.cfg.cfg_target == "uncond":
            out[:half] = self._apply_injection(out[:half])
        elif self.cfg.cfg_target == "cond":
            out[half:] = self._apply_injection(out[half:])
        else:
            raise ValueError(f"Unknown cfg_target: {self.cfg.cfg_target}")

        return out

    def _apply_injection(self, tensor: torch.Tensor) -> torch.Tensor:
        direction = self._prepare_direction(tensor.shape[1], tensor.device, tensor.dtype)
        step_scale = self._step_scale()
        strength = float(self.cfg.strength) * float(step_scale)
        
        if strength == 0.0:
            return tensor

        if self.cfg.injection_style == "broadcast":
            # The highly effective spatial pasting method
            update_spatial = direction.view(1, -1, 1, 1)
            
            if self.cfg.normalize == "rms":
                sample_scale = tensor.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
                delta = strength * sample_scale * update_spatial
            else:
                delta = strength * update_spatial
                
            return tensor + delta

        elif self.cfg.injection_style == "projected":
            # The more subtle, directional projection method
            summary = tensor.mean(dim=(2, 3)) 
            coeff = torch.matmul(summary, direction) 
            
            if self.cfg.normalize == "rms":
                sample_scale = tensor.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
            else:
                sample_scale = torch.ones_like(coeff)

            delta = strength * sample_scale
            update = delta[:, None, None, None] * direction[None, :, None, None]

            if self.cfg.target_delta_ratio > 0:
                tensor_rms = tensor.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
                update_rms = update.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
                target_rms = tensor_rms * float(self.cfg.target_delta_ratio)
                ratio = target_rms / update_rms
                auto_scale = torch.minimum(torch.ones_like(ratio), ratio).clamp(max=float(self.cfg.max_auto_scale))
                update = update * auto_scale[:, None, None, None]

            return tensor + update
            
        return tensor

    def _maybe_log_effect(self, original: torch.Tensor, modified: torch.Tensor) -> None:
        if self.cfg.log_every <= 0:
            return
        if self.current_step % self.cfg.log_every != 0:
            return

        with torch.no_grad():
            delta = modified - original
            base_rms = original.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
            delta_rms = delta.pow(2).mean(dim=(1, 2, 3)).sqrt()
            ratio = (delta_rms / base_rms).mean().item()
            log(f"[injector] step={self.current_step:02d} effective_delta_ratio={ratio:.4f}")


# =========================================================
# Activation capture & Generation
# =========================================================

class SDXLActivationExtractor:
    def __init__(self, model_id: str = "stabilityai/stable-diffusion-xl-base-1.0", enable_cpu_offload: bool = False):
        self.device = get_default_device()
        self.dtype = get_default_dtype(self.device)

        load_kwargs = {"torch_dtype": self.dtype, "use_safetensors": True}
        if self.device == "cuda" and self.dtype == torch.float16:
            load_kwargs["variant"] = "fp16"

        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)
        
        # AGGRESSIVE VRAM PROTECTIONS
        self.pipe.enable_vae_slicing()

        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            log("[init] Enabled CPU offload")
        else:
            self.pipe = self.pipe.to(self.device)

    def _resolve_target_module(self, module_path: str) -> torch.nn.Module:
        parts = module_path.split(".")
        obj = self.pipe
        for part in parts:
            if part.isdigit(): obj = obj[int(part)]
            else: obj = getattr(obj, part)
        return obj

    @torch.inference_mode()
    def capture_timestep_vectors(self, prompt: str, seed: int, cfg: CaptureConfig) -> TraceResult:
        target_module = self._resolve_target_module(cfg.module_path)
        captured_vectors: List[torch.Tensor] = []

        def pool_activation(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[0] >= 2 and tensor.shape[0] % 2 == 0 and cfg.guidance_scale > 1.0:
                tensor = tensor[tensor.shape[0] // 2:]

            mean_vec = tensor.mean(dim=(0, 2, 3)) if tensor.dim() == 4 else tensor.mean(dim=(0, 1))
            if cfg.use_mean_std_pool:
                std_vec = tensor.std(dim=(0, 2, 3) if tensor.dim() == 4 else (0, 1), unbiased=False)
                vec = torch.cat([mean_vec, std_vec], dim=0)
            else:
                vec = mean_vec
            return normalize_vec(vec.detach().float().cpu())

        def hook_fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                captured_vectors.append(pool_activation(tensor))

        handle = target_module.register_forward_hook(hook_fn)

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self.pipe(
                prompt=prompt, num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale, height=cfg.height, width=cfg.width,
                generator=generator,
            )
            return TraceResult(prompt=prompt, seed=seed, timestep_vectors=captured_vectors)
        finally:
            handle.remove()

    @torch.inference_mode()
    def generate_with_injection(self, prompt: str, seed: int, cfg: CaptureConfig, injector: Optional[ActivationInjector] = None):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        if injector: injector.install(self.pipe)
        try:
            result = self.pipe(
                prompt=prompt, num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale, height=cfg.height, width=cfg.width,
                generator=generator,
            )
            image = result.images[0]
            
            # Prevent OOM between generation loops
            del result
            gc.collect()
            torch.cuda.empty_cache()
            
            return image
        finally:
            if injector: injector.uninstall()


# =========================================================
# Vector Discovery (Probe & Mean Delta)
# =========================================================

def make_gender_counterfactuals(prompt: str) -> Tuple[str, str]:
    lower = prompt.lower()
    for article in [" an ", " a "]:
        if article in lower:
            idx = lower.find(article)
            pos = prompt[:idx + len(article)] + "male " + prompt[idx + len(article):]
            neg = prompt[:idx + len(article)] + "female " + prompt[idx + len(article):]
            return pos, neg
    return "male " + prompt, "female " + prompt


def fit_linear_probe_torch(X: torch.Tensor, y: torch.Tensor, lr: float = 1e-2, epochs: int = 500) -> ProbeFitResult:
    X, y = X.float(), y.float()
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-4)

    for _ in range(epochs):
        loss = F.binary_cross_entropy_with_logits(X @ w + b, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = (torch.sigmoid(X @ w + b) >= 0.5).long()
        acc = (preds == y.long()).float().mean().item()

    return ProbeFitResult(weight=w.detach().clone(), bias=float(b.item()), train_accuracy=acc)


def discover_vectors(extractor, cfg, professions, seeds) -> VectorDiscoveryResult:
    pos_traces, neg_traces = [], []
    
    for prof in tqdm(professions, desc="Collecting traces"):
        target = f"A realistic profile picture of a {prof}"
        pos_prompt, neg_prompt = make_gender_counterfactuals(target)
        for seed in seeds:
            pos_traces.append(extractor.capture_timestep_vectors(pos_prompt, seed, cfg))
            neg_traces.append(extractor.capture_timestep_vectors(neg_prompt, seed, cfg))

    x_list, y_list = [], []
    for t in pos_traces:
        x_list.append(normalize_vec(torch.stack(t.timestep_vectors).mean(dim=0)))
        y_list.append(1)
    for t in neg_traces:
        x_list.append(normalize_vec(torch.stack(t.timestep_vectors).mean(dim=0)))
        y_list.append(0)

    X, y = torch.stack(x_list, dim=0), torch.tensor(y_list, dtype=torch.float32)
    probe = fit_linear_probe_torch(X, y)
    
    deltas = [normalize_vec(torch.stack(p.timestep_vectors).mean(dim=0)) - 
              normalize_vec(torch.stack(n.timestep_vectors).mean(dim=0)) 
              for p, n in zip(pos_traces, neg_traces)]
    mean_delta = normalize_vec(torch.stack(deltas).mean(dim=0))

    return VectorDiscoveryResult(
        probe_weight=normalize_vec(probe.weight.float()),
        probe_bias=probe.bias,
        probe_train_accuracy=probe.train_accuracy,
        mean_delta_vector=mean_delta
    )


def maybe_flip_direction(
    extractor: SDXLActivationExtractor,
    cfg: CaptureConfig,
    steering_vector: torch.Tensor,
    test_profession: str = "Doctor",
    seed: int = 0,
) -> torch.Tensor:
    male_prompt = f"A realistic profile picture of a male {test_profession}"
    female_prompt = f"A realistic profile picture of a female {test_profession}"

    male_trace = extractor.capture_timestep_vectors(male_prompt, seed, cfg)
    female_trace = extractor.capture_timestep_vectors(female_prompt, seed, cfg)

    male_vec = normalize_vec(torch.stack(male_trace.timestep_vectors).mean(dim=0))
    female_vec = normalize_vec(torch.stack(female_trace.timestep_vectors).mean(dim=0))
    direction = normalize_vec(steering_vector.float())

    male_score = torch.dot(male_vec, direction).item()
    female_score = torch.dot(female_vec, direction).item()
    log(f"[calibrate] male_score={male_score:.4f} female_score={female_score:.4f}")

    if male_score < female_score:
        log("[calibrate] Flipping steering vector sign so positive = male")
        direction = -direction

    return direction


def load_vectors(vectors_dir: str, professions: List[str], layer: str, preferred_axis: str = "gender") -> torch.Tensor:
    vectors = []
    root = Path(vectors_dir)
    filename = f"layer_{layer}_mean.pt"

    for prof in professions:
        prof_slug = prof.lower().replace(" ", "_")
        prof_dir = root / prof_slug

        if not prof_dir.exists():
            log(f"[load] Missing profession directory: {prof_dir}")
            continue

        axis_candidate = prof_dir / preferred_axis / filename
        if axis_candidate.exists():
            vectors.append(torch.load(axis_candidate, map_location="cpu", weights_only=True))
            log(f"[load] Loaded {axis_candidate}")
            continue

        flat_candidate = prof_dir / filename
        if flat_candidate.exists():
            vectors.append(torch.load(flat_candidate, map_location="cpu", weights_only=True))
            log(f"[load] Loaded {flat_candidate}")
            continue

        nested_matches = sorted(prof_dir.glob(f"*/{filename}"))
        if nested_matches:
            fallback = nested_matches[0]
            vectors.append(torch.load(fallback, map_location="cpu", weights_only=True))
            log(f"[load] Preferred axis '{preferred_axis}' not found for {prof_slug}; using {fallback.parent.name}")
            continue

        log(f"[load] No matching '{filename}' found for profession: {prof_slug}")

    if not vectors:
        raise RuntimeError(f"No vectors found in {vectors_dir} for layer='{layer}' (preferred_axis='{preferred_axis}')")

    return normalize_vec(torch.stack(vectors).mean(dim=0))


# =========================================================
# Main Execution
# =========================================================

DEFAULT_PROFESSIONS = ["Nurse", "Fire Fighter", "Programmer", "Kindergarten Teacher", "Doctor"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--enable-cpu-offload", action="store_true")
    p.add_argument("--module-path", type=str, default="unet.mid_block")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    
    p.add_argument("--mode", type=str, choices=["discover", "load"], default="discover")
    p.add_argument("--load-vectors-dir", type=str, default="")
    p.add_argument("--load-layer", type=str, default="mid_block")
    p.add_argument("--load-axis", type=str, default="gender")
    
    p.add_argument(
        "--inject-layer",
        type=str,
        default="mid_block",
        choices=["mid_block", "down_0", "down_1", "down_2", "up_0", "up_1", "up_2"],
    )
    p.add_argument("--inject-style", type=str, choices=["broadcast", "projected"], default="broadcast")
    p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="both")
    p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
    p.add_argument("--inject-schedule", type=str, choices=["flat", "linear", "cosine", "sine"], default="flat")
    p.add_argument("--inject-log-every", type=int, default=5)
    p.add_argument("--inject-mean-std-strategy", type=str, choices=["truncate", "mean_plus_std"], default="truncate")
    p.add_argument(
        "--inject-target-delta-ratio",
        type=float,
        default=0.0,
        help="RMS(update)/RMS(activation) threshold used by autoscaling. Set 0 to disable.",
    )
    p.add_argument("--calibrate-sign", action="store_true")
    p.add_argument("--calibrate-profession", type=str, default="Doctor")
    p.add_argument("--calibrate-seed", type=int, default=0)
    p.add_argument("--generate-outcomes", action="store_true")
    p.add_argument("--eval-strengths", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    
    p.add_argument("--inject-start-step", type=int, default=0)
    p.add_argument("--inject-end-step", type=int, default=-1, help="Use -1 for no end step limit")
    
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default="outputs/injection_results")
    return p.parse_args()


def main():
    args = parse_args()
    set_global_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    extractor = SDXLActivationExtractor(args.model_id, args.enable_cpu_offload)
    cfg = CaptureConfig(module_path=args.module_path, num_inference_steps=args.steps, guidance_scale=args.guidance)

    if args.mode == "discover":
        log("[vector] Mode: DISCOVER")
        result = discover_vectors(extractor, cfg, DEFAULT_PROFESSIONS, [0, 1, 2])
        steering_vector = result.mean_delta_vector 
        torch.save(steering_vector, os.path.join(args.output_dir, "steering_vector_discovered.pt"))
    else:
        log("[vector] Mode: LOAD")
        steering_vector = load_vectors(
            args.load_vectors_dir,
            DEFAULT_PROFESSIONS,
            args.load_layer,
            args.load_axis,
        )

    if args.calibrate_sign:
        steering_vector = maybe_flip_direction(
            extractor=extractor,
            cfg=cfg,
            steering_vector=steering_vector,
            test_profession=args.calibrate_profession,
            seed=args.calibrate_seed,
        )
        torch.save(steering_vector, os.path.join(args.output_dir, "steering_vector_calibrated.pt"))

    if args.generate_outcomes:
        out_dir = os.path.join(args.output_dir, "outcomes")
        os.makedirs(out_dir, exist_ok=True)
        
        for prof in DEFAULT_PROFESSIONS:
            prompt = f"A realistic profile picture of a {prof}, studio lighting, with the face clearly visible"
            prof_dir = os.path.join(out_dir, prof.lower().replace(" ", "_"))
            os.makedirs(prof_dir, exist_ok=True)

            log(f"[outcomes] Baseline for: {prompt}")
            baseline = extractor.generate_with_injection(prompt, args.seed, cfg)
            baseline.save(os.path.join(prof_dir, "baseline.png"))

            for strength in args.eval_strengths:
                for sign, s_val in [("masculine", strength), ("feminine", -strength)]:
                    # MAP ARGUMENTS TO INJECTOR CONFIG
                    end_step_val = None if args.inject_end_step < 0 else args.inject_end_step
                    inj_cfg = InjectorConfig(
                        layer_key=args.inject_layer,
                        injection_style=args.inject_style,
                        strength=s_val,
                        cfg_target=args.inject_cfg_target,
                        normalize=args.inject_normalize,
                        start_step=args.inject_start_step,
                        end_step=end_step_val,
                        schedule=args.inject_schedule,
                        mean_std_strategy=args.inject_mean_std_strategy,
                        target_delta_ratio=args.inject_target_delta_ratio,
                        log_every=args.inject_log_every,
                    )
                    
                    injector = ActivationInjector(inj_cfg, steering_vector)
                    img = extractor.generate_with_injection(prompt, args.seed, cfg, injector)
                    img.save(os.path.join(prof_dir, f"{sign}_s{str(abs(s_val)).replace('.', 'p')}.png"))

if __name__ == "__main__":
    main()