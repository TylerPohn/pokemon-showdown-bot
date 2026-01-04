# GPU Training Validation Complete - 2025-01-04

## Summary

Successfully debugged and ran decoder-small validation training on AWS GPU (NVIDIA A10G). The training pipeline is now fully functional.

## Issues Fixed

### 1. poke-env Module Structure (0.8 → 0.11)
- **Problem**: `from poke_env.battle import AbstractBattle` failed - module didn't exist in 0.8.3
- **Fix**: Upgraded `poke-env>=0.11` in pyproject.toml (new version has `battle` module)

### 2. NumPy Version Conflict
- **Problem**: poke-env 0.11 requires `numpy>=2.0`, but we had `numpy<2` pinned for PyTorch 2.1
- **Fix**: Upgraded to PyTorch 2.5 + `numpy>=2.0` in Dockerfile and pyproject.toml

### 3. Sequence Padding Bug (`sequence_dataset.py:187-214`)
- **Problem**: `_pad_observations()` was padding the wrong dimension - padded features instead of sequence length
- **Before**: `pad_shape = [0] * (2 * (tensor.dim() - 1)) + [0, pad_len]` → padded last dim
- **After**: For 2D tensors `[seq_len, features]`, use `pad(tensor, (0, 0, 0, pad_len))` to pad first dim

### 4. BCTrainer Nested Dict Handling (`bc_trainer.py:163-170`)
- **Problem**: `batch = {k: v.to(device) for k, v in batch.items()}` failed when `v` was a dict
- **Fix**: Added `_move_to_device()` helper that recursively handles nested dicts

### 5. DecoderPolicy Interface Mismatch
- **Problem**: `DecoderPolicy.forward()` expects `x: [batch, seq_len, d_model]` tensor, but BCTrainer passes a dict
- **Fix**: Created `DecoderPolicyWithEncoder` wrapper class that:
  - Accepts dict observations from SequenceCollator
  - Concatenates `pokemon_features`, `hazards`, and scalars into single tensor
  - Projects from obs_dim (403) to d_model (256)
  - Forwards through DecoderPolicy

## Key Code Changes

| File | Change |
|------|--------|
| `pyproject.toml` | `poke-env>=0.11`, `numpy>=2.0` |
| `Dockerfile` | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |
| `sequence_dataset.py` | Fixed `_pad_observations()` dimension ordering |
| `bc_trainer.py` | Added `_move_to_device()` for nested dicts |
| `decoder_policy.py` | Added `DecoderPolicyWithEncoder` wrapper |
| `factory.py` | Updated to return `DecoderPolicyWithEncoder` for decoder models |

## Training Results

```
Model: DecoderPolicyWithEncoder
Parameters: 3.71M (decoder-small)
GPU: NVIDIA A10G (23.6 GB)
Training speed: ~40 it/s
Status: SUCCESS
```

## Ready for Next Steps

1. **Scale training data** - Current dataset has only 3,721 samples from 200 trajectories
2. **Run medium model** - `decoder-medium` (~50M params) to test scaling
3. **Full 200M training** - `decoder` with wandb tracking

## Commits

- `f391dde` - Fix sequence padding bug
- `660cbf1` - Add DecoderPolicyWithEncoder and fix bc_trainer for nested dicts
- `617f4f9` - Add set_value_head to DecoderPolicyWithEncoder
- `3c8829b` - Upgrade to PyTorch 2.5 and numpy 2 for poke-env 0.11 compatibility
