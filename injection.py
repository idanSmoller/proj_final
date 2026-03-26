#!/usr/bin/env python3
"""
Gender Bias Vector Injection (Improved)

This script discovers gender bias steering vectors and applies them via
directional activation injection into SDXL UNet layers.

Key improvements:
1. Directional steering instead of naive broadcasted feature pasting
2. Better CFG branch handling
3. Smooth timestep scheduling
4. Per-sample normalization instead of global RMS
5. Safer tensor extraction from structured module outputs
6. Optional polarity calibration so positive can consistently mean "male"
7. Outcome generation now defaults to all professions, not just 2 prompts

The script can:
- discover steering vectors from counterfactual prompts
- load precomputed steering vectors
- inject them into SDXL generation for comparison images
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

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
    mode: str = "add"                     # "add", "replace", "blend"
    strength: float = 1.0
    cfg_target: str = "cond"              # "cond", "uncond", "both"
    normalize: str = "rms"                # "none", "rms"
    start_step: int = 0
    end_step: Optional[int] = None
    schedule: str = "flat"                # "flat", "linear", "cosine", "sine"
    blend_alpha: float = 0.5
    mean_std_strategy: str = "truncate"   # "truncate" or "mean_plus_std"
    target_delta_ratio: float = 0.12      # 0 disables autoscaling
    auto_scale_mode: str = "cap"          # "cap" keeps strength effect, "match" reproduces old behavior
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
    losses: List[float]


# =========================================================
# Activation Injector
# =========================================================

class ActivationInjector:
    """
    Directional activation injector for SDXL UNet features.

    Core idea:
    - treat the learned steering vector as a channel-space direction
    - summarize current activation per sample into channel statistics
    - compute its projection on the steering direction
    - modify only that directional component
    """

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
            f"[injector] Installed on {self.cfg.layer_key}, "
            f"mode={self.cfg.mode}, strength={self.cfg.strength}, "
            f"cfg_target={self.cfg.cfg_target}, schedule={self.cfg.schedule}"
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
            if idx >= len(unet.up_blocks):
                raise ValueError(f"up_blocks index {idx} out of range")
            return unet.up_blocks[idx]
        if layer_key.startswith("down_"):
            idx = int(layer_key.split("_")[1])
            if idx >= len(unet.down_blocks):
                raise ValueError(f"down_blocks index {idx} out of range")
            return unet.down_blocks[idx]

        raise ValueError(f"Unknown layer_key: {layer_key}")

    def _hook_fn(self, module, inputs, output):
        if not self._should_inject():
            self.current_step += 1
            return output

        tensor, where = self._extract_main_tensor(output)
        if tensor is None:
            self.current_step += 1
            return output

        if tensor.dim() != 4:
            self.current_step += 1
            return output

        modified = self._apply_injection(tensor)
        self.injected_steps += 1
        self._maybe_log_effect(tensor, modified)
        self.current_step += 1
        return self._reinsert_tensor(output, where, modified)

    def _extract_main_tensor(self, output):
        if torch.is_tensor(output):
            return output, None

        if isinstance(output, tuple):
            for i, item in enumerate(output):
                if torch.is_tensor(item):
                    return item, ("tuple", i)

        if isinstance(output, list):
            for i, item in enumerate(output):
                if torch.is_tensor(item):
                    return item, ("list", i)

        return None, None

    def _reinsert_tensor(self, output, where, new_tensor):
        if where is None:
            return new_tensor

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
        if self.current_step < self.cfg.start_step:
            return False
        if self.cfg.end_step is not None and self.current_step >= self.cfg.end_step:
            return False
        return True

    def _step_scale(self) -> float:
        if not self._should_inject():
            return 0.0

        if self.cfg.end_step is None:
            return 1.0

        start = self.cfg.start_step
        end = self.cfg.end_step
        length = max(end - start, 1)
        pos = self.current_step - start

        if length == 1:
            return 1.0

        t = pos / (length - 1)
        t = max(0.0, min(1.0, t))

        if self.cfg.schedule == "flat":
            return 1.0
        if self.cfg.schedule == "linear":
            return 1.0 - t
        if self.cfg.schedule == "cosine":
            return 0.5 * (1.0 - math.cos(t * math.pi))
        if self.cfg.schedule == "sine":
            return math.sin(t * math.pi)

        raise ValueError(f"Unknown schedule: {self.cfg.schedule}")

    def _prepare_direction(self, tensor: torch.Tensor) -> torch.Tensor:
        direction = self.activation.to(tensor.device, dtype=tensor.dtype)

        if direction.dim() == 4:
            if direction.shape[0] != 1 or direction.shape[2:] != (1, 1):
                raise ValueError(f"Unexpected 4D steering shape: {tuple(direction.shape)}")
            direction = direction[0, :, 0, 0]
        elif direction.dim() == 2:
            if direction.shape[0] != 1:
                raise ValueError(f"Unexpected 2D steering shape: {tuple(direction.shape)}")
            direction = direction[0]
        elif direction.dim() == 1:
            pass
        else:
            raise ValueError(f"Unexpected steering vector shape: {tuple(direction.shape)}")

        c = tensor.shape[1]

        if direction.numel() == 2 * c:
            if self.cfg.mean_std_strategy == "truncate":
                direction = direction[:c]
            elif self.cfg.mean_std_strategy == "mean_plus_std":
                mean_part = direction[:c]
                std_part = direction[c:]
                direction = mean_part + 0.25 * std_part
            else:
                raise ValueError(f"Unknown mean_std_strategy: {self.cfg.mean_std_strategy}")

        if direction.numel() != c:
            raise ValueError(
                f"Channel mismatch: steering has {direction.numel()} channels, tensor has {c}"
            )

        direction = normalize_vec(direction).to(device=tensor.device, dtype=tensor.dtype)
        return direction

    def _apply_injection(self, tensor: torch.Tensor) -> torch.Tensor:
        direction = self._prepare_direction(tensor)

        if self.cfg.cfg_target == "both" or tensor.shape[0] < 2 or tensor.shape[0] % 2 != 0:
            return self._inject_single(tensor, direction)

        half = tensor.shape[0] // 2
        out = tensor.clone()

        if self.cfg.cfg_target == "uncond":
            out[:half] = self._inject_single(out[:half], direction)
        elif self.cfg.cfg_target == "cond":
            out[half:] = self._inject_single(out[half:], direction)
        else:
            raise ValueError(f"Unknown cfg_target: {self.cfg.cfg_target}")

        return out

    def _inject_single(self, tensor: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {tuple(tensor.shape)}")

        direction = direction.to(device=tensor.device, dtype=tensor.dtype)

        step_scale = self._step_scale()
        if step_scale == 0.0:
            return tensor

        strength = float(self.cfg.strength) * float(step_scale)

        summary = tensor.mean(dim=(2, 3))  # [B, C]
        coeff = torch.matmul(summary, direction)  # [B]

        if self.cfg.normalize == "rms":
            sample_scale = tensor.pow(2).mean(dim=(1, 2, 3)).sqrt()
        elif self.cfg.normalize == "none":
            sample_scale = torch.ones_like(coeff)
        else:
            raise ValueError(f"Unknown normalize mode: {self.cfg.normalize}")

        if self.cfg.mode == "add":
            delta = strength * sample_scale
        elif self.cfg.mode == "replace":
            target = strength * sample_scale
            delta = target - coeff
        elif self.cfg.mode == "blend":
            target = strength * sample_scale
            delta = float(self.cfg.blend_alpha) * (target - coeff)
        else:
            raise ValueError(f"Unknown injection mode: {self.cfg.mode}")

        update = delta[:, None, None, None] * direction[None, :, None, None]

        if self.cfg.target_delta_ratio > 0:
            tensor_rms = tensor.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
            update_rms = update.pow(2).mean(dim=(1, 2, 3)).sqrt().clamp_min(1e-8)
            target_rms = tensor_rms * float(self.cfg.target_delta_ratio)

            ratio = target_rms / update_rms
            if self.cfg.auto_scale_mode == "match":
                # Legacy behavior: force delta RMS to match target RMS.
                auto_scale = ratio
            elif self.cfg.auto_scale_mode == "cap":
                # Recommended: only shrink oversized updates, preserve strength otherwise.
                auto_scale = torch.minimum(torch.ones_like(ratio), ratio)
            else:
                raise ValueError(f"Unknown auto_scale_mode: {self.cfg.auto_scale_mode}")

            auto_scale = auto_scale.clamp(max=float(self.cfg.max_auto_scale))
            update = update * auto_scale[:, None, None, None]

        return tensor + update

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
            log(
                f"[injector] step={self.current_step:02d} "
                f"effective_delta_ratio={ratio:.4f}"
            )

    def step_callback_on_step_end(self, pipe, step_idx, timestep, callback_kwargs):
        return callback_kwargs


# =========================================================
# Activation capture
# =========================================================

class SDXLActivationExtractor:
    """Captures activations from SDXL during generation."""
    def __init__(
        self,
        model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
        enable_cpu_offload: bool = False,
    ) -> None:
        self.device = device or get_default_device()
        self.dtype = dtype or get_default_dtype(self.device)

        load_kwargs = {
            "torch_dtype": self.dtype,
            "use_safetensors": True,
        }
        if self.device == "cuda" and self.dtype == torch.float16:
            load_kwargs["variant"] = "fp16"

        log(f"[init] Loading SDXL model='{model_id}' device={self.device} dtype={self.dtype}")
        t0 = time.time()

        self.pipe = StableDiffusionXLPipeline.from_pretrained(model_id, **load_kwargs)

        log(f"[init] Model loaded in {time.time() - t0:.2f}s")

        if enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()
            log("[init] Enabled CPU offload")
        else:
            self.pipe = self.pipe.to(self.device)
            log(f"[init] Pipeline moved to device={self.device}")

        self.pipe.set_progress_bar_config(disable=False)

    def _resolve_target_module(self, module_path: str) -> torch.nn.Module:
        parts = module_path.split(".")
        obj = self.pipe
        for part in parts:
            if part.isdigit():
                obj = obj[int(part)]
            else:
                obj = getattr(obj, part)
        return obj

    @torch.no_grad()
    def capture_timestep_vectors(
        self,
        prompt: str,
        seed: int,
        cfg: CaptureConfig,
    ) -> TraceResult:
        target_module = self._resolve_target_module(cfg.module_path)
        captured_vectors: List[torch.Tensor] = []

        def pool_activation(tensor: torch.Tensor) -> torch.Tensor:
            if tensor.shape[0] >= 2 and tensor.shape[0] % 2 == 0 and cfg.guidance_scale > 1.0:
                half = tensor.shape[0] // 2
                tensor = tensor[half:]

            if tensor.dim() == 4:
                mean_vec = tensor.mean(dim=(0, 2, 3))
                if cfg.use_mean_std_pool:
                    std_vec = tensor.std(dim=(0, 2, 3), unbiased=False)
                    vec = torch.cat([mean_vec, std_vec], dim=0)
                else:
                    vec = mean_vec
            elif tensor.dim() == 3:
                mean_vec = tensor.mean(dim=(0, 1))
                if cfg.use_mean_std_pool:
                    std_vec = tensor.std(dim=(0, 1), unbiased=False)
                    vec = torch.cat([mean_vec, std_vec], dim=0)
                else:
                    vec = mean_vec
            else:
                vec = tensor.flatten().float()

            return normalize_vec(vec.detach().float().cpu())

        def extract_tensor(output):
            if torch.is_tensor(output):
                return output
            if isinstance(output, (tuple, list)):
                for item in output:
                    if torch.is_tensor(item):
                        return item
            return None

        def hook_fn(module, inputs, output):
            tensor = extract_tensor(output)
            if tensor is None:
                return
            pooled = pool_activation(tensor)
            captured_vectors.append(pooled)

        handle = target_module.register_forward_hook(hook_fn)

        try:
            generator = torch.Generator(device=self.device).manual_seed(seed)

            self.pipe(
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
            )

            if not captured_vectors:
                raise RuntimeError(f"No activations captured from {cfg.module_path}")

            return TraceResult(
                prompt=prompt,
                seed=seed,
                timestep_vectors=captured_vectors,
            )

        finally:
            handle.remove()

    def generate_with_injection(
        self,
        prompt: str,
        seed: int,
        cfg: CaptureConfig,
        injector: Optional[ActivationInjector] = None,
    ):
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if injector:
            injector.install(self.pipe)
            log(
                f"[gen] prompt='{prompt}' seed={seed} "
                f"layer={injector.cfg.layer_key} mode={injector.cfg.mode} "
                f"strength={injector.cfg.strength}"
            )

        try:
            result = self.pipe(
                prompt=prompt,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                height=cfg.height,
                width=cfg.width,
                generator=generator,
            )
            return result.images[0]
        finally:
            if injector:
                injector.uninstall()


# =========================================================
# Counterfactual prompt builders
# =========================================================

def make_gender_counterfactuals(prompt: str) -> Tuple[str, str]:
    """
    Returns (male_prompt, female_prompt).
    Positive is male, negative is female.
    """
    prompt = prompt.strip()
    lower = prompt.lower()

    if " a " in lower:
        idx = lower.find(" a ")
        positive = prompt[:idx + 3] + "male " + prompt[idx + 3:]
        negative = prompt[:idx + 3] + "female " + prompt[idx + 3:]
    elif " an " in lower:
        idx = lower.find(" an ")
        positive = prompt[:idx + 4] + "male " + prompt[idx + 4:]
        negative = prompt[:idx + 4] + "female " + prompt[idx + 4:]
    else:
        positive = "male " + prompt
        negative = "female " + prompt

    return positive, negative


# =========================================================
# Linear probe
# =========================================================

def fit_linear_probe_torch(
    X: torch.Tensor,
    y: torch.Tensor,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    epochs: int = 500,
    verbose: bool = False,
) -> ProbeFitResult:
    if X.dim() != 2:
        raise ValueError("X must be [N, D]")
    if y.dim() != 1:
        raise ValueError("y must be [N]")

    X = X.float()
    y = y.float()

    d = X.shape[1]
    w = torch.zeros(d, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    losses: List[float] = []

    log(f"[probe-fit] START X_shape={tuple(X.shape)} y_shape={tuple(y.shape)} epochs={epochs}")

    epoch_iter = range(epochs)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Probe fit", leave=True, mininterval=2.0)

    for _ in epoch_iter:
        logits = X @ w + b
        loss = F.binary_cross_entropy_with_logits(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    with torch.no_grad():
        logits = X @ w + b
        preds = (torch.sigmoid(logits) >= 0.5).long()
        acc = (preds == y.long()).float().mean().item()

    log(f"[probe-fit] END train_accuracy={acc:.4f}")

    return ProbeFitResult(
        weight=w.detach().clone(),
        bias=float(b.item()),
        train_accuracy=acc,
        losses=losses,
    )


def build_probe_dataset_from_traces(
    positive_traces: List[TraceResult],
    negative_traces: List[TraceResult],
    aggregate: str = "mean",
    timestep_index: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    x_list: List[torch.Tensor] = []
    y_list: List[int] = []

    def trace_to_vec(trace: TraceResult) -> torch.Tensor:
        vecs = trace.timestep_vectors
        if aggregate == "mean":
            return normalize_vec(torch.stack(vecs, dim=0).mean(dim=0))
        if aggregate == "single":
            if timestep_index is None:
                raise ValueError("timestep_index required for aggregate='single'")
            if timestep_index < 0 or timestep_index >= len(vecs):
                raise ValueError(f"timestep_index {timestep_index} out of range for {len(vecs)} steps")
            return normalize_vec(vecs[timestep_index])
        raise ValueError(f"Unknown aggregate: {aggregate}")

    for trace in positive_traces:
        x_list.append(trace_to_vec(trace))
        y_list.append(1)

    for trace in negative_traces:
        x_list.append(trace_to_vec(trace))
        y_list.append(0)

    X = torch.stack(x_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.float32)
    return X, y


# =========================================================
# Vector loading
# =========================================================

def load_identification_vectors_all_professions(
    vectors_dir: str,
    professions: List[str],
    layer: str,
    axis: Optional[str] = None,
) -> torch.Tensor:
    vectors = []
    checked_paths: List[str] = []

    def candidate_paths(vectors_root: Path, prof_slug: str) -> List[Path]:
        if axis:
            return [
                vectors_root / prof_slug / axis / f"layer_{layer}_mean.pt",
                vectors_root / prof_slug / f"layer_{layer}_mean.pt",
            ]
        return [vectors_root / prof_slug / f"layer_{layer}_mean.pt"]

    vectors_root = Path(vectors_dir)

    for prof in professions:
        prof_slug = prof.lower().replace(" ", "_")
        loaded = False

        for candidate in candidate_paths(vectors_root, prof_slug):
            checked_paths.append(str(candidate))
            if candidate.exists():
                vec = torch.load(candidate, map_location="cpu")
                vectors.append(vec)
                log(f"[load] Loaded vector from {candidate}")
                loaded = True
                break

        if not loaded:
            if axis:
                log(f"[load] Warning: No vector found for {prof} (axis='{axis}'), skipping")
            else:
                log(f"[load] Warning: No vector found for {prof}, skipping")

    if not vectors:
        sample_checked = "\n  - ".join(checked_paths[:10])
        if len(checked_paths) > 10:
            sample_checked += "\n  - ..."
        raise RuntimeError(
            "No vectors could be loaded. Checked paths include:\n"
            f"  - {sample_checked}"
        )

    mean_vec = normalize_vec(torch.stack(vectors).mean(dim=0))
    log(f"[load] Averaged {len(vectors)} profession vectors")
    return mean_vec


# =========================================================
# Polarity calibration
# =========================================================

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


# =========================================================
# Main experiment
# =========================================================

DEFAULT_PROFESSIONS = [
    "Nurse",
    "Fire Fighter",
    "Programmer",
    "Receptionist",
    "Teacher",
    "Doctor",
    # "Lawyer",
    # "Engineer",
    # "Scientist",
    # "Accountant",
    # "Chef",
    # "Architect",
    # "CEO",
    # "Pilot",
    # "Flight Attendant",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Gender bias vector discovery and injection for SDXL")

    # Model config
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--enable-cpu-offload", action="store_true")

    # Capture config
    p.add_argument("--module-path", type=str, default="unet.mid_block")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--use-mean-std-pool", action="store_true")

    # Vector discovery
    p.add_argument("--mode", type=str, choices=["discover", "load"], default="discover")
    p.add_argument("--professions", type=str, nargs="+", default=DEFAULT_PROFESSIONS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--aggregate", type=str, choices=["mean", "single"], default="mean")
    p.add_argument("--timestep-index", type=int, default=10)
    p.add_argument("--probe-epochs", type=int, default=500)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)

    # Vector loading
    p.add_argument(
        "--load-vectors-dir",
        type=str,
        default="../proj_final/generated_images_final/steering_vectors",
        help="Path to steering_vectors dir from identification.py",
    )
    p.add_argument("--load-layer", type=str, default="mid_block")
    p.add_argument("--load-axis", type=str, default="gender")

    # Injection config
    p.add_argument(
        "--inject-layer",
        type=str,
        default="mid_block",
        choices=["mid_block", "down_0", "down_1", "down_2", "up_0", "up_1", "up_2"],
        help="Layer to inject into",
    )
    p.add_argument("--inject-mode", type=str, choices=["add", "replace", "blend"], default="add")
    p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="cond")
    p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
    p.add_argument("--inject-start-step", type=int, default=0)
    p.add_argument("--inject-end-step", type=int, default=-1, help="Use -1 to inject until final step")
    p.add_argument("--inject-schedule", type=str, choices=["flat", "linear", "cosine", "sine"], default="flat")
    p.add_argument("--inject-blend-alpha", type=float, default=0.5)
    p.add_argument(
        "--inject-target-delta-ratio",
        type=float,
        default=0.12,
        help="RMS(update)/RMS(activation) threshold used by autoscaling. Set 0 to disable.",
    )
    p.add_argument(
        "--inject-auto-scale-mode",
        type=str,
        choices=["cap", "match"],
        default="cap",
        help="'cap' preserves strength differences; 'match' forces all strengths toward the same delta RMS.",
    )
    p.add_argument("--inject-max-auto-scale", type=float, default=80.0)
    p.add_argument("--inject-log-every", type=int, default=5)
    p.add_argument(
        "--inject-mean-std-strategy",
        type=str,
        choices=["truncate", "mean_plus_std"],
        default="truncate",
    )

    # Calibration
    p.add_argument("--calibrate-sign", action="store_true")
    p.add_argument("--calibrate-profession", type=str, default="Doctor")
    p.add_argument("--calibrate-seed", type=int, default=0)

    # Generation
    p.add_argument(
        "--eval-prompts",
        type=str,
        nargs="+",
        default=None,
        help="If omitted, prompts are built automatically from --professions.",
    )
    p.add_argument("--generate-outcomes", action="store_true")
    p.add_argument("--eval-strengths", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--eval-seed", type=int, default=1234)

    # Output
    p.add_argument("--output-dir", type=str, default="outputs/injection_results")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = SDXLActivationExtractor(
        model_id=args.model_id,
        enable_cpu_offload=args.enable_cpu_offload,
    )

    cfg = CaptureConfig(
        module_path=args.module_path,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        height=args.height,
        width=args.width,
        use_mean_std_pool=args.use_mean_std_pool,
    )

    # -----------------------------------------------------
    # Get steering vector
    # -----------------------------------------------------
    if args.mode == "discover":
        log("[vector] Mode: DISCOVER")

        positive_traces: List[TraceResult] = []
        negative_traces: List[TraceResult] = []

        for profession in tqdm(args.professions, desc="Collecting traces"):
            target = f"A realistic profile picture of a {profession}"
            positive, negative = make_gender_counterfactuals(target)

            for seed in args.seeds:
                pos_trace = extractor.capture_timestep_vectors(positive, seed, cfg)
                neg_trace = extractor.capture_timestep_vectors(negative, seed, cfg)
                positive_traces.append(pos_trace)
                negative_traces.append(neg_trace)

        X, y = build_probe_dataset_from_traces(
            positive_traces=positive_traces,
            negative_traces=negative_traces,
            aggregate=args.aggregate,
            timestep_index=args.timestep_index if args.aggregate == "single" else None,
        )

        probe = fit_linear_probe_torch(
            X=X,
            y=y,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            epochs=args.probe_epochs,
            verbose=True,
        )

        steering_vector = normalize_vec(probe.weight.float())
        log(f"[vector] Discovered vector, probe_accuracy={probe.train_accuracy:.4f}")

        vector_path = os.path.join(args.output_dir, "steering_vector.pt")
        torch.save(steering_vector.cpu(), vector_path)
        log(f"[save] Saved discovered vector to {vector_path}")

    elif args.mode == "load":
        log("[vector] Mode: LOAD")

        steering_vector = load_identification_vectors_all_professions(
            vectors_dir=args.load_vectors_dir,
            professions=args.professions,
            layer=args.load_layer,
            axis=args.load_axis,
        )
        log(f"[vector] Loaded averaged vector for layer {args.load_layer}")

        vector_path = os.path.join(args.output_dir, "steering_vector_loaded.pt")
        torch.save(steering_vector.cpu(), vector_path)
        log(f"[save] Saved loaded vector snapshot to {vector_path}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # -----------------------------------------------------
    # Optional sign calibration
    # -----------------------------------------------------
    if args.calibrate_sign:
        steering_vector = maybe_flip_direction(
            extractor=extractor,
            cfg=cfg,
            steering_vector=steering_vector,
            test_profession=args.calibrate_profession,
            seed=args.calibrate_seed,
        )
        calibrated_path = os.path.join(args.output_dir, "steering_vector_calibrated.pt")
        torch.save(steering_vector.cpu(), calibrated_path)
        log(f"[save] Saved calibrated vector to {calibrated_path}")

    # -----------------------------------------------------
    # Build eval prompts
    # -----------------------------------------------------
    if args.eval_prompts is None:
        eval_prompts = [f"A realistic profile picture of a {p}" for p in args.professions]
        log(f"[eval] Using {len(eval_prompts)} auto-generated prompts from professions")
    else:
        eval_prompts = args.eval_prompts
        log(f"[eval] Using {len(eval_prompts)} explicit eval prompts")

    # -----------------------------------------------------
    # Outcome generation
    # -----------------------------------------------------
    if args.generate_outcomes:
        log("[outcomes] Generating comparison images")

        outcomes_dir = os.path.join(args.output_dir, "outcomes")
        os.makedirs(outcomes_dir, exist_ok=True)

        for prompt_idx, prompt in enumerate(eval_prompts):
            prompt_tag = slugify(prompt, max_len=60)
            prompt_dir = os.path.join(outcomes_dir, f"{prompt_idx:02d}_{prompt_tag}")
            os.makedirs(prompt_dir, exist_ok=True)

            log(f"[outcomes] Baseline for: {prompt}")
            baseline = extractor.generate_with_injection(prompt, args.eval_seed, cfg, injector=None)
            baseline.save(os.path.join(prompt_dir, "baseline.png"))

            for strength in args.eval_strengths:
                for sign_name, signed_strength in [("masculine", strength), ("feminine", -strength)]:
                    log(
                        f"[outcomes] prompt='{prompt}' sign={sign_name} "
                        f"strength={abs(signed_strength)}"
                    )

                    injector_cfg = InjectorConfig(
                        layer_key=args.inject_layer,
                        mode=args.inject_mode,
                        strength=signed_strength,
                        cfg_target=args.inject_cfg_target,
                        normalize=args.inject_normalize,
                        start_step=args.inject_start_step,
                        end_step=None if args.inject_end_step < 0 else args.inject_end_step,
                        schedule=args.inject_schedule,
                        blend_alpha=args.inject_blend_alpha,
                        mean_std_strategy=args.inject_mean_std_strategy,
                        target_delta_ratio=args.inject_target_delta_ratio,
                        auto_scale_mode=args.inject_auto_scale_mode,
                        max_auto_scale=args.inject_max_auto_scale,
                        log_every=args.inject_log_every,
                    )

                    injector = ActivationInjector(cfg=injector_cfg, activation=steering_vector)
                    result = extractor.generate_with_injection(prompt, args.eval_seed, cfg, injector)

                    safe_strength = str(abs(signed_strength)).replace(".", "p")
                    filename = f"{sign_name}_s{safe_strength}.png"
                    result.save(os.path.join(prompt_dir, filename))

        log(f"[outcomes] Saved to {outcomes_dir}")

    log("[main] DONE")


if __name__ == "__main__":
    main()