from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import torch

from sdxl_counterfactual_bias_metrics import (
    CaptureConfig,
    SDXLActivationExtractor,
    SDXLBiasExperiment,
    TraceResult,
    build_probe_dataset_from_traces,
    fit_linear_probe_torch,
    log,
    make_gender_counterfactuals,
    normalize_vec,
    set_global_seed,
)


INJECTION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Injection"))
if INJECTION_DIR not in sys.path:
    sys.path.insert(0, INJECTION_DIR)

from sdxl_injector import ActivationInjector, InjectorConfig


@dataclass
class VectorDiscoveryResult:
    probe_weight: torch.Tensor
    probe_bias: float
    probe_train_accuracy: float
    mean_delta_vector: torch.Tensor
    per_timestep_delta: torch.Tensor
    probe_pairwise_accuracy: float
    mean_delta_pairwise_accuracy: float


def _default_eval_prompts() -> List[str]:
    return [
        "A realistic profile picture of a Doctor",
        "A realistic profile picture of a Nurse",
        "A realistic profile picture of a Scientist",
        "A realistic profile picture of a CEO",
    ]


def _default_professions() -> List[str]:
    return [
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


def _trace_to_aggregate_vec(trace: TraceResult, aggregate: str, timestep_index: int) -> torch.Tensor:
    if aggregate == "mean":
        return normalize_vec(torch.stack(trace.timestep_vectors, dim=0).mean(dim=0))
    if aggregate == "single":
        return normalize_vec(trace.timestep_vectors[timestep_index])
    raise ValueError("aggregate must be one of {'mean', 'single'}")


def _pairwise_direction_accuracy(
    pos_traces: Sequence[TraceResult],
    neg_traces: Sequence[TraceResult],
    direction: torch.Tensor,
    aggregate: str,
    timestep_index: int,
) -> float:
    correct = 0
    total = 0

    d = normalize_vec(direction.float())

    for pos, neg in zip(pos_traces, neg_traces):
        pos_vec = _trace_to_aggregate_vec(pos, aggregate=aggregate, timestep_index=timestep_index)
        neg_vec = _trace_to_aggregate_vec(neg, aggregate=aggregate, timestep_index=timestep_index)
        if float(torch.dot(pos_vec, d)) > float(torch.dot(neg_vec, d)):
            correct += 1
        total += 1

    return float(correct / max(total, 1))


def _compute_mean_delta_vector(
    pos_traces: Sequence[TraceResult],
    neg_traces: Sequence[TraceResult],
    aggregate: str,
    timestep_index: int,
) -> torch.Tensor:
    deltas: List[torch.Tensor] = []
    for pos, neg in zip(pos_traces, neg_traces):
        pos_vec = _trace_to_aggregate_vec(pos, aggregate=aggregate, timestep_index=timestep_index)
        neg_vec = _trace_to_aggregate_vec(neg, aggregate=aggregate, timestep_index=timestep_index)
        deltas.append(pos_vec - neg_vec)

    stacked = torch.stack(deltas, dim=0)
    return normalize_vec(stacked.mean(dim=0))


def _compute_per_timestep_delta(
    pos_traces: Sequence[TraceResult],
    neg_traces: Sequence[TraceResult],
) -> torch.Tensor:
    if not pos_traces:
        raise ValueError("No positive traces available")
    n_steps = len(pos_traces[0].timestep_vectors)

    deltas: List[torch.Tensor] = []
    for t in range(n_steps):
        step_deltas: List[torch.Tensor] = []
        for pos, neg in zip(pos_traces, neg_traces):
            step_deltas.append(pos.timestep_vectors[t] - neg.timestep_vectors[t])
        deltas.append(normalize_vec(torch.stack(step_deltas, dim=0).mean(dim=0)))

    return torch.stack(deltas, dim=0)


def discover_gender_bias_vector(
    exp: SDXLBiasExperiment,
    professions: Sequence[str],
    seeds: Sequence[int],
    aggregate: str,
    timestep_index: int,
    probe_epochs: int,
    probe_lr: float,
    probe_weight_decay: float,
) -> VectorDiscoveryResult:
    pos_traces: List[TraceResult] = []
    neg_traces: List[TraceResult] = []

    log(
        f"[vector] START professions={list(professions)} seeds={list(seeds)} "
        f"aggregate={aggregate} timestep_index={timestep_index}"
    )

    for profession in professions:
        target = f"A realistic profile picture of a {profession}"
        positive, negative = make_gender_counterfactuals(target)

        for seed in seeds:
            _, pos_trace, neg_trace = exp.capture_triplet(
                target_prompt=target,
                positive_prompt=positive,
                negative_prompt=negative,
                seed=int(seed),
                profession=None,
                positive_label="male",
                negative_label="female",
            )
            pos_traces.append(pos_trace)
            neg_traces.append(neg_trace)

    dataset_aggregate = "mean" if aggregate == "mean" else "single"
    ts_index = timestep_index if aggregate == "single" else None

    X, y = build_probe_dataset_from_traces(
        positive_traces=pos_traces,
        negative_traces=neg_traces,
        aggregate=dataset_aggregate,
        timestep_index=ts_index,
    )

    probe = fit_linear_probe_torch(
        X=X,
        y=y,
        lr=probe_lr,
        weight_decay=probe_weight_decay,
        epochs=probe_epochs,
        verbose=True,
    )

    probe_weight = normalize_vec(probe.weight.float())
    mean_delta = _compute_mean_delta_vector(
        pos_traces=pos_traces,
        neg_traces=neg_traces,
        aggregate=aggregate,
        timestep_index=timestep_index,
    )
    per_timestep_delta = _compute_per_timestep_delta(pos_traces=pos_traces, neg_traces=neg_traces)

    probe_pair_acc = _pairwise_direction_accuracy(
        pos_traces=pos_traces,
        neg_traces=neg_traces,
        direction=probe_weight,
        aggregate=aggregate,
        timestep_index=timestep_index,
    )
    mean_delta_pair_acc = _pairwise_direction_accuracy(
        pos_traces=pos_traces,
        neg_traces=neg_traces,
        direction=mean_delta,
        aggregate=aggregate,
        timestep_index=timestep_index,
    )

    log(
        "[vector] END "
        f"probe_train_acc={probe.train_accuracy:.4f} "
        f"probe_pair_acc={probe_pair_acc:.4f} "
        f"mean_delta_pair_acc={mean_delta_pair_acc:.4f}"
    )

    return VectorDiscoveryResult(
        probe_weight=probe_weight,
        probe_bias=float(probe.bias),
        probe_train_accuracy=float(probe.train_accuracy),
        mean_delta_vector=mean_delta,
        per_timestep_delta=per_timestep_delta,
        probe_pairwise_accuracy=probe_pair_acc,
        mean_delta_pairwise_accuracy=mean_delta_pair_acc,
    )


def _to_injection_activation(vec: torch.Tensor, expected_channels: int) -> torch.Tensor:
    v = vec.detach().float().cpu().flatten()

    if v.numel() == expected_channels * 2:
        # If vector comes from mean+std pooling, keep the mean subspace for spatial injection.
        v = v[:expected_channels]
    elif v.numel() != expected_channels:
        raise ValueError(
            f"Vector dim mismatch for injection: got {v.numel()} values, expected {expected_channels} or {expected_channels * 2}."
        )

    return normalize_vec(v).view(1, expected_channels, 1, 1)


def _run_pipe_with_optional_step_callback(pipe, call_kwargs: Dict[str, object], injector: ActivationInjector) -> object:
    try:
        return pipe(
            **call_kwargs,
            callback_on_step_end=injector.step_callback_on_step_end,
        )
    except TypeError:
        return pipe(**call_kwargs)


def generate_outcome_images(
    extractor: SDXLActivationExtractor,
    cfg: CaptureConfig,
    direction_vec: torch.Tensor,
    prompts: Sequence[str],
    strengths: Sequence[float],
    seed: int,
    out_dir: str,
    mode: str,
    cfg_target: str,
    normalize: str,
    start_step: int,
    end_step: int,
    layer_idx: int,
) -> None:
    if not hasattr(extractor.pipe, "unet") or not hasattr(extractor.pipe.unet, "down_blocks"):
        raise ValueError("Pipeline U-Net structure is unexpected; cannot resolve injection target layer.")

    down_blocks = extractor.pipe.unet.down_blocks
    if layer_idx < 0 or layer_idx >= len(down_blocks):
        raise ValueError(f"inject-layer-idx out of range: {layer_idx} (down_blocks={len(down_blocks)})")

    block = down_blocks[layer_idx]
    if not hasattr(block, "resnets") or len(block.resnets) == 0:
        raise ValueError(f"down_blocks[{layer_idx}] has no resnets; cannot infer output channels.")

    out_channels = getattr(block.resnets[0], "out_channels", None)
    if out_channels is None:
        raise ValueError(
            f"Cannot infer out_channels from down_blocks[{layer_idx}].resnets[0]."
        )

    act = _to_injection_activation(direction_vec, int(out_channels))
    os.makedirs(out_dir, exist_ok=True)

    for prompt_idx, prompt in enumerate(prompts):
        prompt_tag = prompt.lower().replace(" ", "_").replace("/", "_")
        prompt_tag = "".join(ch for ch in prompt_tag if ch.isalnum() or ch in "_-")[:100]
        prompt_dir = os.path.join(out_dir, f"{prompt_idx:02d}_{prompt_tag}")
        os.makedirs(prompt_dir, exist_ok=True)

        base_gen = torch.Generator(device=extractor.device).manual_seed(int(seed))
        base_result = extractor.pipe(
            prompt=prompt,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            height=cfg.height,
            width=cfg.width,
            generator=base_gen,
        )
        base_path = os.path.join(prompt_dir, "baseline.png")
        base_result.images[0].save(base_path)
        log(f"[outcomes] saved baseline: {base_path}")

        for strength in strengths:
            for sign_name, signed_act in (("plus", act), ("minus", -act)):
                injector = ActivationInjector(
                    cfg=InjectorConfig(
                        layer_idx=int(layer_idx),
                        mode=mode,
                        strength=float(strength),
                        cfg_target=cfg_target,
                        normalize=normalize,
                        start_step=int(start_step),
                        end_step=(None if end_step < 0 else int(end_step)),
                    ),
                    activation=signed_act,
                ).install(extractor.pipe)

                try:
                    gen = torch.Generator(device=extractor.device).manual_seed(int(seed))
                    call_kwargs = {
                        "prompt": prompt,
                        "num_inference_steps": cfg.num_inference_steps,
                        "guidance_scale": cfg.guidance_scale,
                        "height": cfg.height,
                        "width": cfg.width,
                        "generator": gen,
                    }
                    out = _run_pipe_with_optional_step_callback(
                        extractor.pipe,
                        call_kwargs,
                        injector,
                    )
                finally:
                    injector.uninstall()

                safe_strength = str(strength).replace(".", "p")
                out_path = os.path.join(prompt_dir, f"{sign_name}_s{safe_strength}.png")
                out.images[0].save(out_path)
                log(f"[outcomes] saved {sign_name} strength={strength}: {out_path}")


def _write_summary_csv(path: str, rows: List[Dict[str, float]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = [
        "metric",
        "value",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learn male-female bias vectors in SDXL activation space")
    p.add_argument("--model-id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--module-path", type=str, default="unet.mid_block")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=5.0)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--professions", type=str, nargs="+", default=_default_professions())
    p.add_argument("--aggregate", type=str, choices=["mean", "single"], default="mean")
    p.add_argument("--timestep-index", type=int, default=10)
    p.add_argument("--probe-epochs", type=int, default=500)
    p.add_argument("--probe-lr", type=float, default=1e-2)
    p.add_argument("--probe-weight-decay", type=float, default=1e-4)
    p.add_argument("--output-dir", type=str, default="outputs/sdxl_bias_vectors")
    p.add_argument("--use-mean-std-pool", action="store_true")
    p.add_argument("--generate-outcomes", action="store_true")
    p.add_argument("--evaluation-prompts", type=str, nargs="+", default=_default_eval_prompts())
    p.add_argument("--evaluation-strengths", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--evaluation-vector", type=str, choices=["probe", "mean_delta"], default="probe")
    p.add_argument("--evaluation-out-dir", type=str, default="outputs/sdxl_bias_vector_outcomes")
    p.add_argument("--inject-mode", type=str, choices=["add", "replace", "blend"], default="add")
    p.add_argument("--inject-cfg-target", type=str, choices=["cond", "uncond", "both"], default="cond")
    p.add_argument("--inject-normalize", type=str, choices=["none", "rms"], default="rms")
    p.add_argument("--inject-start-step", type=int, default=5)
    p.add_argument("--inject-end-step", type=int, default=20,
                   help="Set -1 for no end-step limit.")
    p.add_argument("--inject-layer-idx", type=int, default=2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--enable-cpu-offload", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(int(args.seed))

    extractor = SDXLActivationExtractor(
        model_id=args.model_id,
        enable_cpu_offload=bool(args.enable_cpu_offload),
    )

    cfg = CaptureConfig(
        module_path=args.module_path,
        num_inference_steps=int(args.steps),
        guidance_scale=float(args.guidance),
        height=int(args.height),
        width=int(args.width),
        use_mean_std_pool=bool(args.use_mean_std_pool),
        debug=False,
        hook_log_every=5,
        show_diffusers_progress=True,
        save_images=False,
    )

    exp = SDXLBiasExperiment(extractor=extractor, cfg=cfg)

    result = discover_gender_bias_vector(
        exp=exp,
        professions=args.professions,
        seeds=args.seeds,
        aggregate=args.aggregate,
        timestep_index=int(args.timestep_index),
        probe_epochs=int(args.probe_epochs),
        probe_lr=float(args.probe_lr),
        probe_weight_decay=float(args.probe_weight_decay),
    )

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    probe_path = os.path.join(out_dir, "male_female_probe_weight.pt")
    mean_delta_path = os.path.join(out_dir, "male_female_mean_delta.pt")
    per_t_path = os.path.join(out_dir, "male_female_per_timestep_delta.pt")
    meta_path = os.path.join(out_dir, "male_female_vector_meta.pt")
    summary_csv = os.path.join(out_dir, "summary.csv")

    torch.save(result.probe_weight.cpu(), probe_path)
    torch.save(result.mean_delta_vector.cpu(), mean_delta_path)
    torch.save(result.per_timestep_delta.cpu(), per_t_path)
    torch.save(
        {
            "probe_bias": result.probe_bias,
            "probe_train_accuracy": result.probe_train_accuracy,
            "probe_pairwise_accuracy": result.probe_pairwise_accuracy,
            "mean_delta_pairwise_accuracy": result.mean_delta_pairwise_accuracy,
            "aggregate": args.aggregate,
            "timestep_index": int(args.timestep_index),
            "module_path": args.module_path,
            "steps": int(args.steps),
            "guidance": float(args.guidance),
            "professions": list(args.professions),
            "seeds": list(args.seeds),
            "probe_weight_path": probe_path,
            "mean_delta_path": mean_delta_path,
            "per_timestep_delta_path": per_t_path,
        },
        meta_path,
    )

    _write_summary_csv(
        summary_csv,
        rows=[
            {"metric": "probe_train_accuracy", "value": result.probe_train_accuracy},
            {"metric": "probe_pairwise_accuracy", "value": result.probe_pairwise_accuracy},
            {"metric": "mean_delta_pairwise_accuracy", "value": result.mean_delta_pairwise_accuracy},
            {"metric": "probe_bias", "value": result.probe_bias},
        ],
    )

    log(f"[save] probe weight: {probe_path}")
    log(f"[save] mean delta: {mean_delta_path}")
    log(f"[save] per-timestep delta: {per_t_path}")
    log(f"[save] metadata: {meta_path}")
    log(f"[save] summary: {summary_csv}")

    if args.generate_outcomes:
        eval_vec = result.probe_weight if args.evaluation_vector == "probe" else result.mean_delta_vector
        log(
            f"[outcomes] START prompts={args.evaluation_prompts} strengths={args.evaluation_strengths} "
            f"vector={args.evaluation_vector}"
        )
        generate_outcome_images(
            extractor=extractor,
            cfg=cfg,
            direction_vec=eval_vec,
            prompts=args.evaluation_prompts,
            strengths=args.evaluation_strengths,
            seed=int(args.seed),
            out_dir=args.evaluation_out_dir,
            mode=args.inject_mode,
            cfg_target=args.inject_cfg_target,
            normalize=args.inject_normalize,
            start_step=int(args.inject_start_step),
            end_step=int(args.inject_end_step),
            layer_idx=int(args.inject_layer_idx),
        )
        log(f"[outcomes] END dir={args.evaluation_out_dir}")


if __name__ == "__main__":
    main()
