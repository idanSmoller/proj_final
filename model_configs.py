#!/usr/bin/env python3
"""
Model Configuration System

Defines architecture-specific configurations for different diffusion models
to enable consistent bias analysis across model families.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple
import torch


@dataclass
class ModelConfig:
    """Configuration for a specific generative model."""
    model_id: str
    model_name: str
    model_family: str  # "sdxl", "sd15", "sd21", "kandinsky"

    # Architecture-specific settings
    target_layer_keys: List[str]
    text_encoder_type: str  # "clip", "dual_clip", "clip_kandinsky"

    # Generation defaults
    default_height: int
    default_width: int
    dtype: Optional[torch.dtype] = None
    variant: Optional[str] = None

    # Text encoding method
    pooled_projection_dim: Optional[int] = None

    # Memory profile (estimated peak VRAM in GB)
    estimated_vram_gb: float = 8.0
    recommended_min_vram_gb: float = 10.0

    def __post_init__(self):
        """Set dtype defaults based on variant."""
        if self.dtype is None:
            if self.variant == "fp16":
                self.dtype = torch.float16
            else:
                self.dtype = torch.bfloat16


# =========================================================
# Model Definitions
# =========================================================

SDXL_BASE_CONFIG = ModelConfig(
    model_id="stabilityai/stable-diffusion-xl-base-1.0",
    model_name="SDXL Base 1.0",
    model_family="sdxl",
    target_layer_keys=["mid_block", "up_0", "up_1", "up_2"],
    text_encoder_type="dual_clip",
    default_height=1024,
    default_width=1024,
    variant="fp16",
    pooled_projection_dim=1280,
    estimated_vram_gb=9.5,
    recommended_min_vram_gb=11.0,
)

SD_1_5_CONFIG = ModelConfig(
    model_id="runwayml/stable-diffusion-v1-5",
    model_name="Stable Diffusion 1.5",
    model_family="sd15",
    target_layer_keys=["mid_block", "up_0", "up_1", "up_2"],
    text_encoder_type="clip",
    default_height=512,
    default_width=512,
    variant="fp16",
    pooled_projection_dim=768,
    estimated_vram_gb=5.5,
    recommended_min_vram_gb=6.0,
)

SD_2_1_CONFIG = ModelConfig(
    model_id="sd2-community/stable-diffusion-2-1",
    model_name="Stable Diffusion 2.1",
    model_family="sd21",
    target_layer_keys=["mid_block", "up_0", "up_1", "up_2"],
    text_encoder_type="clip",
    default_height=768,
    default_width=768,
    variant="fp16",
    pooled_projection_dim=1024,
    estimated_vram_gb=7.5,
    recommended_min_vram_gb=8.0,
)

SD_1_4_CONFIG = ModelConfig(
    model_id="CompVis/stable-diffusion-v1-4",
    model_name="Stable Diffusion 1.4",
    model_family="sd14",
    target_layer_keys=["mid_block", "up_0", "up_1", "up_2"],
    text_encoder_type="clip",
    default_height=512,
    default_width=512,
    variant="fp16",
    pooled_projection_dim=768,
    estimated_vram_gb=5.5,
    recommended_min_vram_gb=6.0,
)

OPENJOURNEY_V4_CONFIG = ModelConfig(
    model_id="prompthero/openjourney-v4",
    model_name="Openjourney v4",
    model_family="sd15",  # Based on SD 1.5
    target_layer_keys=["mid_block", "up_0", "up_1", "up_2"],
    text_encoder_type="clip",
    default_height=512,
    default_width=512,
    variant="fp16",
    pooled_projection_dim=768,
    estimated_vram_gb=5.5,
    recommended_min_vram_gb=6.0,
)


# Registry of available models
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    "sdxl": SDXL_BASE_CONFIG,
    "sd15": SD_1_5_CONFIG,
    "sd21": SD_2_1_CONFIG,
    "sd14": SD_1_4_CONFIG,
    "openjourney": OPENJOURNEY_V4_CONFIG,
}


def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key."""
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_key]


def list_available_models() -> List[str]:
    """List available model keys."""
    return list(MODEL_REGISTRY.keys())


# =========================================================
# Architecture Adapters
# =========================================================

class ArchitectureAdapter:
    """Adapts operations to specific model architectures."""

    @staticmethod
    def get_target_module(unet, layer_key: str):
        """Get module by layer key consistently across architectures."""
        if layer_key == "mid_block":
            return unet.mid_block
        if layer_key.startswith("up_"):
            idx = int(layer_key.split("_")[1])
            return unet.up_blocks[idx]
        if layer_key.startswith("down_"):
            idx = int(layer_key.split("_")[1])
            return unet.down_blocks[idx]
        raise ValueError(f"Unknown layer key: {layer_key}")

    @staticmethod
    def get_text_embeddings(pipe, prompt: str, model_config: ModelConfig) -> torch.Tensor:
        """Extract text embeddings based on model family."""
        with torch.no_grad():
            if model_config.model_family == "sdxl":
                # SDXL has dual text encoders with pooled output
                _, _, pooled_embeds, _ = pipe.encode_prompt(
                    prompt,
                    device=pipe.device,
                    do_classifier_free_guidance=False
                )
                return pooled_embeds[0].float().cpu()

            elif model_config.model_family in ["sd15", "sd21"]:
                # SD 1.5/2.1 use single CLIP encoder
                # Get the pooled text embedding
                text_inputs = pipe.tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids.to(pipe.device)

                # Get encoder outputs
                encoder_output = pipe.text_encoder(text_input_ids)

                # For SD models, use the pooled output (last hidden state of [EOS] token)
                # or extract the [CLS] token embedding
                if hasattr(encoder_output, 'pooler_output') and encoder_output.pooler_output is not None:
                    pooled = encoder_output.pooler_output
                else:
                    # Use the last hidden state's first token (CLS)
                    pooled = encoder_output.last_hidden_state[:, 0, :]

                return pooled[0].float().cpu()

            else:
                raise ValueError(f"Unsupported model family: {model_config.model_family}")

    @staticmethod
    def load_pipeline(model_config: ModelConfig, device: str = "cuda"):
        """Load appropriate pipeline for model family."""
        from diffusers import DiffusionPipeline
        import gc

        torch.cuda.empty_cache()
        gc.collect()

        load_kwargs = {
            "torch_dtype": model_config.dtype,
        }

        if model_config.variant:
            load_kwargs["variant"] = model_config.variant

        pipe = DiffusionPipeline.from_pretrained(
            model_config.model_id,
            **load_kwargs
        ).to(torch.device(device))

        pipe.enable_model_cpu_offload()

        return pipe


# =========================================================
# Helper Functions
# =========================================================

def get_model_info(model_key: str) -> str:
    """Get formatted info string for a model."""
    config = get_model_config(model_key)
    return (
        f"{config.model_name}\n"
        f"  ID: {config.model_id}\n"
        f"  Family: {config.model_family}\n"
        f"  Resolution: {config.default_width}x{config.default_height}\n"
        f"  Layers: {', '.join(config.target_layer_keys)}\n"
        f"  Est. VRAM: ~{config.estimated_vram_gb:.1f} GB\n"
        f"  Recommended: {config.recommended_min_vram_gb:.0f}+ GB\n"
    )


def print_all_models():
    """Print info for all available models."""
    print("Available Models:")
    print("=" * 60)
    for key in list_available_models():
        print(f"\nKey: {key}")
        print(get_model_info(key))
    print("=" * 60)


def check_gpu_compatibility(model_key: str, available_vram_gb: float) -> Tuple[bool, str]:
    """Check if model is compatible with available VRAM."""
    config = get_model_config(model_key)

    if available_vram_gb >= config.recommended_min_vram_gb:
        return True, f"✅ {model_key}: Safe ({available_vram_gb:.0f}GB >= {config.recommended_min_vram_gb:.0f}GB recommended)"
    elif available_vram_gb >= config.estimated_vram_gb:
        return True, f"⚠️  {model_key}: Tight fit ({available_vram_gb:.0f}GB, needs ~{config.estimated_vram_gb:.1f}GB)"
    else:
        return False, f"❌ {model_key}: May fail ({available_vram_gb:.0f}GB < {config.estimated_vram_gb:.1f}GB needed)"


def get_safe_models_for_vram(available_vram_gb: float) -> List[str]:
    """Get list of models that should work with available VRAM."""
    safe_models = []
    for model_key in list_available_models():
        config = get_model_config(model_key)
        if available_vram_gb >= config.estimated_vram_gb:
            safe_models.append(model_key)
    return safe_models


if __name__ == "__main__":
    import sys

    print_all_models()

    # Check GPU compatibility
    if torch.cuda.is_available():
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n{'='*60}")
        print(f"GPU Detected: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {vram_gb:.1f} GB")
        print(f"\nCompatibility Check:")
        print(f"{'='*60}")

        all_safe = True
        for model_key in list_available_models():
            compatible, msg = check_gpu_compatibility(model_key, vram_gb)
            print(msg)
            if not compatible:
                all_safe = False

        print(f"{'='*60}")

        if all_safe:
            print("✅ All models should work with your GPU!")
        else:
            safe = get_safe_models_for_vram(vram_gb)
            print(f"\n⚠️  Recommended models for your GPU: {', '.join(safe)}")
    else:
        print("\n⚠️  No CUDA GPU detected")

