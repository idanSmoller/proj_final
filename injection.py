#!/usr/bin/env python3
"""
Gender Bias Vector Injection (Enhanced)

This script discovers gender bias steering vectors and applies them via sophisticated
injection mechanisms. It can either:
1. Discover vectors using linear probe + mean delta methods
2. Load pre-computed vectors from identification.py output
3. Apply vectors with configurable injection strategies

Enhanced from sdxl_bias_vector_discovery to support user's layer structure.
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    return F.normalize(x.unsqueeze(0), p=2, dim=1).squeeze(0)


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
    layer_key: str = "mid_block"  # "mid_block", "up_0", "up_1", "up_2"
    mode: str = "add"  # "add", "replace", "blend"
    strength: float = 1.0
    cfg_target: str = "cond"  # "cond", "uncond", "both"
    normalize: str = "rms"  # "none", "rms"
    start_step: int = 5
    end_step: Optional[int] = 20


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
# Activation Injector (Enhanced for multi-layer support)
# =========================================================

class ActivationInjector:
    """
    Injects steering vectors into SDXL UNet activations with sophisticated controls.
    Supports mid_block and up_blocks (not just down_blocks).
    """
    def __init__(self, cfg: InjectorConfig, activation: torch.Tensor):
        self.cfg = cfg
        self.activation = activation
        self.current_step = 0
        self.handle = None
        self.target_module = None

    def install(self, pipe: StableDiffusionXLPipeline) -> "ActivationInjector":
        """Attach hook to the target layer."""
        self.target_module = self._resolve_target_module(pipe)
        self.handle = self.target_module.register_forward_hook(self._hook_fn)
        self.current_step = 0
        log(f"[injector] Installed on {self.cfg.layer_key}, mode={self.cfg.mode}, strength={self.cfg.strength}")
        return self

    def uninstall(self) -> None:
        """Remove hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            log("[injector] Uninstalled")

    def _resolve_target_module(self, pipe: StableDiffusionXLPipeline) -> torch.nn.Module:
        """Resolve layer_key to actual module."""
        unet = pipe.unet
        layer_key = self.cfg.layer_key

        if layer_key == "mid_block":
            return unet.mid_block
        elif layer_key.startswith("up_"):
            idx = int(layer_key.split("_")[1])
            if idx >= len(unet.up_blocks):
                raise ValueError(f"up_blocks index {idx} out of range")
            return unet.up_blocks[idx]
        elif layer_key.startswith("down_"):
            idx = int(layer_key.split("_")[1])
            if idx >= len(unet.down_blocks):
                raise ValueError(f"down_blocks index {idx} out of range")
            return unet.down_blocks[idx]
        else:
            raise ValueError(f"Unknown layer_key: {layer_key}")

    def _hook_fn(self, module, inputs, output):
        """Forward hook that injects the activation."""
        # Check if we should inject at this step
        if not self._should_inject():
            self.current_step += 1
            return output

        # Get the tensor to modify
        if torch.is_tensor(output):
            tensor = output
        elif isinstance(output, (tuple, list)):
            # Find the largest tensor (main activation)
            tensors = [item for item in output if torch.is_tensor(item)]
            if not tensors:
                self.current_step += 1
                return output
            tensor = max(tensors, key=lambda t: t.numel())
        else:
            self.current_step += 1
            return output

        # Apply injection
        modified = self._apply_injection(tensor)

        # Replace in output
        if torch.is_tensor(output):
            output = modified
        elif isinstance(output, tuple):
            output_list = list(output)
            # Find and replace the tensor we modified
            for i, item in enumerate(output_list):
                if torch.is_tensor(item) and item is tensor:
                    output_list[i] = modified
                    break
            output = tuple(output_list)
        elif isinstance(output, list):
            for i, item in enumerate(output):
                if torch.is_tensor(item) and item is tensor:
                    output[i] = modified
                    break

        self.current_step += 1
        return output

    def _should_inject(self) -> bool:
        """Check if we should inject at current step."""
        if self.current_step < self.cfg.start_step:
            return False
        if self.cfg.end_step is not None and self.current_step >= self.cfg.end_step:
            return False
        return True

    def _apply_injection(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply the injection to the tensor."""
        # Prepare activation vector
        act = self.activation.to(tensor.device, dtype=tensor.dtype)

        # Handle activation shape - ensure it matches tensor channels
        if act.dim() == 1:
            # [C] -> [1, C, 1, 1]
            act = act.view(1, -1, 1, 1)
        elif act.dim() == 2:
            # [1, C] -> [1, C, 1, 1]
            act = act.view(1, -1, 1, 1)
        elif act.dim() == 4:
            # Already [1, C, 1, 1]
            pass
        else:
            raise ValueError(f"Unexpected activation shape: {act.shape}")

        # Ensure channel dimension matches
        if act.shape[1] != tensor.shape[1]:
            # If activation has 2x channels (mean+std pooling), take first half
            if act.shape[1] == tensor.shape[1] * 2:
                act = act[:, :tensor.shape[1]]
            else:
                raise ValueError(f"Channel mismatch: activation={act.shape[1]}, tensor={tensor.shape[1]}")

        # Handle CFG batching (tensor might be [2, C, H, W] with [uncond, cond])
        if tensor.shape[0] == 2 and self.cfg.cfg_target != "both":
            if self.cfg.cfg_target == "cond":
                # Only modify conditional branch
                tensor = tensor.clone()
                tensor[1:2] = self._inject_single(tensor[1:2], act)
            elif self.cfg.cfg_target == "uncond":
                # Only modify unconditional branch
                tensor = tensor.clone()
                tensor[0:1] = self._inject_single(tensor[0:1], act)
            return tensor
        else:
            # Apply to all
            return self._inject_single(tensor, act)

    def _inject_single(self, tensor: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        """Apply injection to tensor with specified mode."""
        strength = self.cfg.strength

        # Normalize if requested
        if self.cfg.normalize == "rms":
            tensor_rms = tensor.pow(2).mean().sqrt()
            act = act / (act.pow(2).mean().sqrt() + 1e-8) * tensor_rms

        # Apply mode
        if self.cfg.mode == "add":
            return tensor + (strength * act)
        elif self.cfg.mode == "replace":
            return strength * act.expand_as(tensor)
        elif self.cfg.mode == "blend":
            return strength * act.expand_as(tensor) + (1 - strength) * tensor
        else:
            raise ValueError(f"Unknown injection mode: {self.cfg.mode}")

    def step_callback_on_step_end(self, pipe, step_idx, timestep, callback_kwargs):
        """Callback compatible with diffusers callback_on_step_end."""
        # This is called after each denoising step
        # We track steps in the hook, so nothing to do here
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
        """Resolve module path like 'unet.mid_block'."""
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
        """Capture activation vectors at each timestep."""
        target_module = self._resolve_target_module(cfg.module_path)
        captured_vectors: List[torch.Tensor] = []

        def pool_activation(tensor: torch.Tensor) -> torch.Tensor:
            """Pool activation to single vector."""
            # Select conditional branch if CFG
            if tensor.shape[0] == 2 and cfg.guidance_scale > 1.0:
                tensor = tensor[1:2]

            if tensor.dim() == 4:
                # [B, C, H, W]
                mean_vec = tensor.mean(dim=(2, 3))
                if cfg.use_mean_std_pool:
                    std_vec = tensor.std(dim=(2, 3), unbiased=False)
                    vec = torch.cat([mean_vec, std_vec], dim=1)
                else:
                    vec = mean_vec
            elif tensor.dim() == 3:
                # [B, T, C]
                mean_vec = tensor.mean(dim=1)
                if cfg.use_mean_std_pool:
                    std_vec = tensor.std(dim=1, unbiased=False)
                    vec = torch.cat([mean_vec, std_vec], dim=1)
                else:
                    vec = mean_vec
            else:
                vec = tensor.flatten(start_dim=1)

            return vec.squeeze(0).detach().float().cpu()

        def hook_fn(module, inputs, output):
            if torch.is_tensor(output):
                tensor = output
            elif isinstance(output, (tuple, list)):
                tensors = [item for item in output if torch.is_tensor(item)]
                if not tensors:
                    return
                tensor = max(tensors, key=lambda t: t.numel())
            else:
                return

            pooled = pool_activation(tensor)
            captured_vectors.append(normalize_vec(pooled))

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
        """Generate image with optional injection."""
        generator = torch.Generator(device=self.device).manual_seed(seed)

        if injector:
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
            return result.images[0]
        finally:
            if injector:
                injector.uninstall()


# =========================================================
# Counterfactual prompt builders
# =========================================================

def make_gender_counterfactuals(prompt: str) -> Tuple[str, str]:
    """Returns (male_prompt, female_prompt)."""
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
    """Fits logistic regression probe."""
    if X.dim() != 2:
        raise ValueError("X must be [N, D]")
    if y.dim() != 1:
        raise ValueError("y must be [N]")

    X = X.float()
    y = y.float()

    D = X.shape[1]
    w = torch.zeros(D, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)
    losses: List[float] = []

    log(f"[probe-fit] START X_shape={tuple(X.shape)} y_shape={tuple(y.shape)} epochs={epochs}")

    epoch_iter = range(epochs)
    if verbose:
        epoch_iter = tqdm(epoch_iter, desc="Probe fit", leave=True, mininterval=2.0)

    for epoch in epoch_iter:
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
    """Build dataset for probe training."""
    X_list: List[torch.Tensor] = []
    y_list: List[int] = []

    def trace_to_vec(trace: TraceResult) -> torch.Tensor:
        vecs = trace.timestep_vectors
        if aggregate == "mean":
            return normalize_vec(torch.stack(vecs, dim=0).mean(dim=0))
        elif aggregate == "single":
            if timestep_index is None:
                raise ValueError("timestep_index required for aggregate='single'")
            return normalize_vec(vecs[timestep_index])
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

    for trace in positive_traces:
        X_list.append(trace_to_vec(trace))
        y_list.append(1)

    for trace in negative_traces:
        X_list.append(trace_to_vec(trace))
        y_list.append(0)

    X = torch.stack(X_list, dim=0)
    y = torch.tensor(y_list, dtype=torch.float32)

    return X, y


# =========================================================
# Vector loading from identification.py output
# =========================================================

def load_identification_vectors(vectors_dir: str, profession: str, layer: str) -> torch.Tensor:
    """
    Load steering vector from identification.py output.

    Args:
        vectors_dir: Path to VECTORS_ROOT from identification.py
        profession: Profession name (e.g., "Nurse")
        layer: Layer key (e.g., "mid_block", "up_0")

    Returns:
        Mean steering vector across timesteps for that layer
    """
    from pathlib import Path

    prof_slug = profession.lower().replace(" ", "_")
    vector_path = Path(vectors_dir) / prof_slug / f"layer_{layer}_mean.pt"

    if not vector_path.exists():
        raise FileNotFoundError(f"Vector not found: {vector_path}")

    vec = torch.load(vector_path, map_location="cpu")
    log(f"[load] Loaded vector from {vector_path}")
    return vec


def load_identification_vectors_all_professions(
    vectors_dir: str,
    professions: List[str],
    layer: str
) -> torch.Tensor:
    """Load and average vectors across all professions."""
    vectors = []
    for prof in professions:
        try:
            vec = load_identification_vectors(vectors_dir, prof, layer)
            vectors.append(vec)
        except FileNotFoundError:
            log(f"[load] Warning: No vector found for {prof}, skipping")

    if not vectors:
        raise RuntimeError("No vectors could be loaded")

    # Average and normalize
    mean_vec = torch.stack(vectors).mean(dim=0)
    mean_vec = normalize_vec(mean_vec)

    log(f"[load] Averaged {len(vectors)} profession vectors")
    return mean_vec


# =========================================================
# Main experiment
# =========================================================

# Standardized profession list (consistent with identification script)
DEFAULT_PROFESSIONS = [
    "Nurse",
    "Fire Fighter",
    "Programmer",
    "Receptionist",
    "Teacher",
    "Doctor",
    "Lawyer",
    "Engineer",
    "Scientist",
    "Accountant",
    "Chef",
    "Architect",
    "CEO",
    "Pilot",
    "Flight Attendant",
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
    p.add_argument("--mode", type=str, choices=["discover", "load"], default="discover",
                   help="'discover' trains new vectors, 'load' uses identification.py output")
    p.add_argument("--professions", type=str, nargs="+", default=DEFAULT_PROFESSIONS)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--aggregate", type=str, choices=["mean", "single"], default="mean")
    p.add_argument("--timestep-index", type=int, default=10)
    p.add_argument("--probe-epochs", type=int, default=500)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)

    # Vector loading (if mode=load)
    p.add_argument("--load-vectors-dir", type=str, default="../proj_final/generated_images_final/steering_vectors",
                   help="Path to steering_vectors dir from identification.py")
    p.add_argument("--load-layer", type=str, default="mid_block",
                   help="Which layer's vectors to load")

    # Injection config
    p.add_argument("--inject-layer", type=str, default="mid_block",
                   choices=["mid_block", "up_0", "up_1", "up_2"],
                   help="Layer to inject into")
    p.add_argument("--inject-mode", type=str, choices=["add", "replace", "blend"], default="add")
    p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="cond")
    p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
    p.add_argument("--inject-start-step", type=int, default=5)
    p.add_argument("--inject-end-step", type=int, default=20)

    # Generation
    p.add_argument("--generate-outcomes", action="store_true",
                   help="Generate comparison images with different injection strengths")
    p.add_argument("--eval-prompts", type=str, nargs="+",
                   default=["A realistic profile picture of a Doctor",
                           "A realistic profile picture of a CEO"])
    p.add_argument("--eval-strengths", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    p.add_argument("--eval-seed", type=int, default=1234)

    # Output
    p.add_argument("--output-dir", type=str, default="outputs/injection_results")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

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

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover or load steering vector
    if args.mode == "discover":
        log("[vector] Mode: DISCOVER - Training probe to find steering vectors")

        # Collect traces
        positive_traces = []
        negative_traces = []

        for profession in tqdm(args.professions, desc="Collecting traces"):
            target = f"A realistic profile picture of a {profession}"
            positive, negative = make_gender_counterfactuals(target)

            for seed in args.seeds:
                pos_trace = extractor.capture_timestep_vectors(positive, seed, cfg)
                neg_trace = extractor.capture_timestep_vectors(negative, seed, cfg)

                positive_traces.append(pos_trace)
                negative_traces.append(neg_trace)

        # Build dataset
        X, y = build_probe_dataset_from_traces(
            positive_traces=positive_traces,
            negative_traces=negative_traces,
            aggregate=args.aggregate,
            timestep_index=args.timestep_index if args.aggregate == "single" else None,
        )

        # Fit probe
        probe = fit_linear_probe_torch(
            X=X,
            y=y,
            lr=args.probe_lr,
            weight_decay=args.probe_weight_decay,
            epochs=args.probe_epochs,
            verbose=True,
        )

        steering_vector = normalize_vec(probe.weight.float())
        log(f"[vector] Discovered vector, probe accuracy={probe.train_accuracy:.4f}")

        # Save
        vector_path = os.path.join(args.output_dir, "steering_vector.pt")
        torch.save(steering_vector.cpu(), vector_path)
        log(f"[save] Saved to {vector_path}")

    elif args.mode == "load":
        log("[vector] Mode: LOAD - Loading from identification.py output")

        steering_vector = load_identification_vectors_all_professions(
            vectors_dir=args.load_vectors_dir,
            professions=args.professions,
            layer=args.load_layer,
        )

        log(f"[vector] Loaded averaged vector for layer {args.load_layer}")

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    # Generate outcome images if requested
    if args.generate_outcomes:
        log("[outcomes] Generating comparison images")

        outcomes_dir = os.path.join(args.output_dir, "outcomes")
        os.makedirs(outcomes_dir, exist_ok=True)

        for prompt_idx, prompt in enumerate(args.eval_prompts):
            prompt_tag = prompt.lower().replace(" ", "_")[:50]
            prompt_dir = os.path.join(outcomes_dir, f"{prompt_idx:02d}_{prompt_tag}")
            os.makedirs(prompt_dir, exist_ok=True)

            # Baseline (no injection)
            log(f"[outcomes] Generating baseline for: {prompt}")
            baseline = extractor.generate_with_injection(prompt, args.eval_seed, cfg, injector=None)
            baseline.save(os.path.join(prompt_dir, "baseline.png"))

            # Injections at different strengths
            for strength in args.eval_strengths:
                for sign_name, scale in [("masculine", strength), ("feminine", -strength)]:
                    log(f"[outcomes] Generating {sign_name} strength={abs(scale)}")

                    injector_cfg = InjectorConfig(
                        layer_key=args.inject_layer,
                        mode=args.inject_mode,
                        strength=scale,
                        cfg_target=args.inject_cfg_target,
                        normalize=args.inject_normalize,
                        start_step=args.inject_start_step,
                        end_step=args.inject_end_step,
                    )

                    injector = ActivationInjector(cfg=injector_cfg, activation=steering_vector)

                    result = extractor.generate_with_injection(prompt, args.eval_seed, cfg, injector)

                    safe_strength = str(abs(scale)).replace(".", "p")
                    filename = f"{sign_name}_s{safe_strength}.png"
                    result.save(os.path.join(prompt_dir, filename))

        log(f"[outcomes] Saved to {outcomes_dir}")

    log("[main] DONE")


if __name__ == "__main__":
    main()
