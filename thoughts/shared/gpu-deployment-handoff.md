# GPU Deployment Handoff - 200M Decoder Training

## Status: VALIDATION COMPLETE ✅

`decoder-small` training completed successfully on NVIDIA A10G GPU!

## What's Been Done

1. **Code synced to GPU** - Latest code pulled from GitHub
2. **PyTorch 2.5 + CUDA 12.4** - Upgraded for numpy 2 / poke-env 0.11 compatibility
3. **Docker image built** - `poke-trainer` with all dependencies
4. **CUDA verified** - `CUDA: True, Device: NVIDIA A10G`
5. **Validation training complete** - decoder-small (3.71M params), 1 epoch, ~40 it/s

## Issues Fixed During Deployment

1. **poke-env import error** - Upgraded to poke-env>=0.11 (new module structure)
2. **NumPy conflict** - Upgraded to PyTorch 2.5 + numpy>=2.0 (poke-env requires numpy 2)
3. **Sequence padding bug** - Fixed `_pad_observations` to pad first dimension correctly
4. **BCTrainer nested dict handling** - Added `_move_to_device` helper for nested observation dicts
5. **DecoderPolicy interface** - Created `DecoderPolicyWithEncoder` wrapper to convert dict->tensor

## GPU Connection

```bash
# Set AWS credentials (get from .env)
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...

# Check command status
aws ssm get-command-invocation \
  --command-id "COMMAND_ID" \
  --instance-id "i-0d37b84d08727a481" \
  --region us-east-1
```

## Training Commands

### Small model (validation)
```bash
cd /home/ec2-user/pokemon-showdown-bot && docker run --gpus all \
  -v /home/ec2-user/pokemon-showdown-bot/data:/app/data \
  -v /home/ec2-user/pokemon-showdown-bot/checkpoints:/app/checkpoints \
  poke-trainer python -m poke.training.train_decoder \
  --model-size decoder-small \
  --epochs 1 \
  --data-path /app/data/trajectories/trajectories.jsonl \
  --batch-size 16
```

### Medium model (~50M params)
```bash
docker run --gpus all \
  -v /home/ec2-user/pokemon-showdown-bot/data:/app/data \
  -v /home/ec2-user/pokemon-showdown-bot/checkpoints:/app/checkpoints \
  poke-trainer python -m poke.training.train_decoder \
  --model-size decoder-medium \
  --epochs 5 \
  --data-path /app/data/trajectories/trajectories.jsonl
```

### Full 200M model
```bash
docker run --gpus all \
  -v /home/ec2-user/pokemon-showdown-bot/data:/app/data \
  -v /home/ec2-user/pokemon-showdown-bot/checkpoints:/app/checkpoints \
  poke-trainer python -m poke.training.train_decoder \
  --model-size decoder \
  --epochs 10 \
  --data-path /app/data/trajectories/trajectories.jsonl \
  --use-wandb
```

## Key Files

- Project: `/home/ec2-user/pokemon-showdown-bot/`
- Training data: `data/trajectories/trajectories.jsonl` (200 trajectories, 3,721 samples)
- Checkpoints: `checkpoints/`
- Docker image: `poke-trainer`

## Model Sizes Reference

| Size | Params | model_type | Notes |
|------|--------|------------|-------|
| Small | ~3.7M | `decoder-small` | For validation/testing |
| Medium | ~50M | `decoder-medium` | Faster iteration |
| Large | ~200M | `decoder` | Production model |

## Running SSM Commands

Template:
```bash
aws ssm send-command \
  --instance-ids "i-0d37b84d08727a481" \
  --document-name "AWS-RunShellScript" \
  --parameters 'commands=["YOUR_COMMAND_HERE"]' \
  --region us-east-1 \
  --timeout-seconds 600
```

## Next Steps

1. **Scale training data** - Need more trajectories for meaningful training
2. **Run medium model** - Test scaling before full 200M
3. **Enable wandb** - Track training metrics
4. **Full training** - Run decoder (200M) with proper hyperparameters
