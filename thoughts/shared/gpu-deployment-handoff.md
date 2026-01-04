# GPU Deployment Handoff - 200M Decoder Training

## Status: TRAINING IN PROGRESS

Validation training with `decoder-small` is currently running on the GPU.

## What's Been Done

1. **Code synced to GPU** - Latest code pulled from GitHub
2. **Dockerfile created** with numpy<2 pin for PyTorch 2.1 compatibility
3. **Docker image built** - `poke-trainer` image ready
4. **CUDA verified** - `CUDA: True, Device: NVIDIA A10G`
5. **Training started** - decoder-small validation run (1 epoch)

## GPU Connection

```bash
# Set AWS credentials
(get them from .env, do not put them in markdown files)

# Check training status (replace COMMAND_ID with current one)
aws ssm get-command-invocation \
  --command-id "8c3afb98-ac99-4655-bd0e-e226df76150c" \
  --instance-id "i-0d37b84d08727a481" \
  --region us-east-1
```

## Current Training Command

```bash
cd /home/ec2-user/pokemon-showdown-bot && docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  poke-trainer python -m poke.training.train_decoder \
  --model-size decoder-small \
  --epochs 1 \
  --data-path /app/data/trajectories/trajectories.jsonl \
  --batch-size 16
```

## Next Steps After Validation

Once decoder-small completes successfully:

1. **Scale to medium model**:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  poke-trainer python -m poke.training.train_decoder \
  --model-size decoder-medium \
  --epochs 5 \
  --data-path /app/data/trajectories/trajectories.jsonl
```

2. **Full 200M model training**:
```bash
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/checkpoints:/app/checkpoints \
  poke-trainer python -m poke.training.train_decoder \
  --model-size decoder \
  --epochs 10 \
  --data-path /app/data/trajectories/trajectories.jsonl \
  --use-wandb
```

## Key Files on GPU

- Project: `/home/ec2-user/pokemon-showdown-bot/`
- Training data: `data/trajectories/trajectories.jsonl` (200 trajectories, 3,721 samples)
- Checkpoints: `checkpoints/`
- Docker image: `poke-trainer`

## Running SSM Commands

Template for running commands on GPU:
```bash
aws ssm send-command \
  --instance-ids "i-0d37b84d08727a481" \
  --document-name "AWS-RunShellScript" \
  --parameters commands='["YOUR_COMMAND_HERE"]' \
  --region us-east-1 \
  --timeout-seconds 600
```

## Issues Fixed

- **NumPy 2.x incompatibility**: Fixed by pinning `numpy>=1.24,<2` in pyproject.toml (commit d63ee34)
- Docker image now uses cached numpy 1.26.0 from base PyTorch image

## Model Sizes Reference

| Size | Params | model_type |
|------|--------|------------|
| Small | ~3.6M | `decoder-small` |
| Medium | ~50M | `decoder-medium` |
| Large | ~200M | `decoder` |
