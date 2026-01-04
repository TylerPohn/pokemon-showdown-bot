# 200M Decoder Model Deployment

## Implementation Status: READY FOR GPU

All code changes have been made and pushed to GitHub. Ready for GPU deployment.

**Repository:** https://github.com/TylerPohn/pokemon-showdown-bot

---

## Task List

### 1. Sync Code to GPU
- [x] Push changes to git
- [ ] SSH into GPU: `aws ssm start-session --target i-0d37b84d08727a481`
- [ ] Switch user: `sudo su - ec2-user`
- [ ] Pull latest code: `git pull origin master`

### 2. Build Docker Image on GPU
- [ ] Create Dockerfile (template below)
- [ ] Build Docker image: `docker build -t poke-trainer .`
- [ ] Verify CUDA is accessible in container

### 3. Prepare Training Data
- [ ] Ensure battle replay data is available on GPU at `data/trajectories.jsonl`
- [ ] Data format: JSONL with `steps` array containing observations/actions

### 4. Run Training
- [ ] Validate with small model: `python -m poke.training.train_decoder --model-size decoder-small --epochs 1`
- [ ] Scale up to medium: `python -m poke.training.train_decoder --model-size decoder-medium`
- [ ] Full training: `python -m poke.training.train_decoder --model-size decoder --epochs 10`

### 5. Validate Model
- [ ] Check loss curves (wandb or logs)
- [ ] Run inference test battles
- [ ] Compare against baseline agents

---

## Files Changed (Synced via Git)

**New Files:**
- `src/poke/models/decoder_policy.py` - Main 200M transformer with RoPE, SwiGLU, Flash Attention
- `src/poke/models/value_head.py` - HL-Gauss value classification head
- `src/poke/training/sequence_dataset.py` - Sequence data loading for transformers
- `src/poke/training/train_decoder.py` - Training script with CLI args

**Modified Files:**
- `src/poke/models/config.py` - ScaledEncoderConfig + SMALL/MEDIUM/LARGE presets
- `src/poke/models/state_encoder.py` - ScaledStateEncoder (14 tokens per turn)
- `src/poke/models/factory.py` - decoder/decoder-small/decoder-medium model types
- `src/poke/training/bc_trainer.py` - AMP + gradient accumulation support
- `src/poke/agents/nn_agent.py` - DecoderAgent class with KV cache

---

## Quick Reference

### Model Sizes
| Size | Params | model_type |
|------|--------|------------|
| Small | ~15M | `decoder-small` |
| Medium | ~50M | `decoder-medium` |
| Large | ~200M | `decoder` or `decoder-large` |

### Create Model
```python
from poke.models.factory import create_policy, get_model_info

policy = create_policy("decoder")  # 200M params
print(get_model_info(policy))
```

### Training Config
```python
from poke.training.bc_trainer import BCConfig

config = BCConfig(
    batch_size=32,
    gradient_accumulation_steps=8,  # effective batch = 256
    use_amp=True,
    amp_dtype="bfloat16",  # or "float16"
    lr=1e-4,
    lr_scheduler="cosine",
    warmup_steps=1000,
)
```

### GPU Memory Estimate (A10 24GB)
- Model: ~800MB (FP16)
- Optimizer states: ~1.6GB
- Gradients: ~800MB
- Activations: ~1-2GB (with checkpointing)
- **Total: ~4-5GB** - plenty of headroom

---

## Dockerfile Template

Create this file on the GPU machine:

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e .

# Install wandb for tracking (optional)
RUN pip install --no-cache-dir wandb

# Default command
CMD ["python", "-m", "poke.training.train_decoder"]
```

---

## GPU Commands

```bash
# 1. Connect to GPU
aws ssm start-session --target i-0d37b84d08727a481
sudo su - ec2-user

# 2. Clone/pull repository
git clone https://github.com/TylerPohn/pokemon-showdown-bot.git
cd pokemon-showdown-bot
# OR if already cloned:
cd pokemon-showdown-bot && git pull origin master

# 3. Create Dockerfile (paste template above)
vim Dockerfile

# 4. Build Docker image
docker build -t poke-trainer .

# 5. Verify CUDA access
docker run --gpus all poke-trainer python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"

# 6. Run training (validation)
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    poke-trainer python -m poke.training.train_decoder \
    --model-size decoder-small \
    --epochs 1 \
    --data-path /app/data/trajectories.jsonl

# 7. Run full training
docker run --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/checkpoints:/app/checkpoints \
    poke-trainer python -m poke.training.train_decoder \
    --model-size decoder \
    --epochs 10 \
    --data-path /app/data/trajectories.jsonl \
    --use-wandb

# Monitor GPU usage
nvidia-smi -l 1
```

---

## Training Script CLI Reference

```bash
python -m poke.training.train_decoder [OPTIONS]

Model:
  --model-size {decoder-small,decoder-medium,decoder}  Model size (default: decoder)

Data:
  --data-path PATH          Training data JSONL (default: data/trajectories.jsonl)
  --val-data-path PATH      Validation data (optional)
  --seq-len INT             Max sequence length (default: 50)
  --max-samples INT         Limit samples for debugging

Training:
  --batch-size INT          Batch size per GPU (default: 32)
  --gradient-accumulation   Accumulation steps (default: 8, effective_batch=256)
  --epochs INT              Training epochs (default: 10)
  --lr FLOAT                Learning rate (default: 1e-4)
  --warmup-steps INT        LR warmup steps (default: 1000)

Mixed Precision:
  --no-amp                  Disable AMP
  --amp-dtype {float16,bfloat16}  AMP dtype (default: bfloat16)

Checkpointing:
  --checkpoint-dir PATH     Checkpoint directory
  --resume PATH             Resume from checkpoint
  --save-every INT          Save every N epochs

Logging:
  --use-wandb               Enable wandb logging
  --wandb-project NAME      Wandb project name
  --log-every INT           Log every N steps
```
