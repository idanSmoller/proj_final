# Multi-Model Gender Bias Analysis - Quick Start

## 🚀 Quick Commands

### Check GPU Compatibility
```bash
python model_configs.py
```

### Test Model Access
```bash
python test_model_access.py
```

### Run Analysis (Quick Mode)
```bash
# Single model (fastest, ~30 min)
python multi_model_runner.py --models sd15 --quick

# Two models (~1 hour)
python multi_model_runner.py --models sd15 sd21 --quick

# All available models
python multi_model_runner.py --models sd15 sd21 sd14 openjourney --quick
```

### Run Full Analysis (16 samples/group, several hours)
```bash
python multi_model_runner.py --models sd15 sd21
```

### Compare Results
```bash
python compare_models.py --models sd15 sd21
```

## 📊 Available Models

| Key | Model | VRAM | Auth Needed? |
|-----|-------|------|--------------|
| `sd15` | Stable Diffusion 1.5 | 5.5 GB | No |
| `sd14` | Stable Diffusion 1.4 | 5.5 GB | No |
| `sd21` | Stable Diffusion 2.1 | 7.5 GB | Maybe* |
| `openjourney` | Openjourney v4 | 5.5 GB | No |
| `sdxl` | SDXL Base 1.0 | 9.5 GB | Maybe* |

*Run `python test_model_access.py` to check

## 🔐 Authentication (If Needed)

If models require authentication:

```bash
# Get token from: https://huggingface.co/settings/tokens
python hf_login.py
```

Or use:
```bash
~/.local/bin/huggingface-cli login
```

## 📁 Output Structure

```
multi_model_results/
├── sd15/
│   ├── bias_analysis.csv          # Bias scores per layer/timestep
│   ├── text_bias_analysis.csv     # Text embedding bias
│   ├── images/                    # Generated images
│   ├── steering_vectors/          # Bias direction vectors
│   └── bias_heatmaps/            # Visualizations
└── sd21/ ...

comparison_results/
├── model_summary.png              # Cross-model statistics
├── text_bias_comparison.png       # Text bias comparison
└── by_profession/                # Per-profession comparisons
```

## 🎯 Recommended Workflow

```bash
# 1. Check what works on your GPU
python model_configs.py
python test_model_access.py

# 2. Quick test
python multi_model_runner.py --models sd15 --quick

# 3. Full analysis
python multi_model_runner.py --models sd15 sd21

# 4. Compare results
python compare_models.py --models sd15 sd21
```

## 💡 Tips

- **Start with `--quick`** to test everything works (~30 min)
- **Use `sd15`** first - it's fastest and most reliable
- **11GB GPU**: Safe with sd15, sd14, sd21; tight with sdxl
- Results are **cached** - interrupted runs can be resumed

## 📊 Understanding Results

**Bias Scores:**
- `+1.0` = Strong masculine bias
- `0.0` = Balanced/neutral
- `-1.0` = Strong feminine bias

**Example:**
```
Doctor   | +0.42 | Masculine bias
Nurse    | -0.58 | Feminine bias
Engineer | +0.31 | Masculine bias
```

## 🆘 Troubleshooting

**Out of memory?**
- Use smaller models: `--models sd15`
- Fewer professions: `--professions "Doctor" "Nurse"`
- Close other GPU applications

**Model not found?**
- Run `python test_model_access.py`
- Check authentication: `python hf_login.py`

**Already running?**
- Cached results are reused automatically
- Delete cache to regenerate: `rm -rf multi_model_results/sd15/activations/`
