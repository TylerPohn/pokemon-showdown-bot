---
date: 2026-01-04T16:30:46Z
researcher: Claude Code
git_commit: HEAD (uncommitted)
branch: master
repository: poke
topic: "Analysis of Most Advanced Battling Approach and Improvement Opportunities"
tags: [research, codebase, battle-ai, offline-rl, neural-networks, improvements]
status: complete
last_updated: 2026-01-04
last_updated_by: Claude Code
---

# Research: Analysis of Most Advanced Battling Approach and Improvement Opportunities

**Date**: 2026-01-04T16:30:46Z
**Researcher**: Claude Code
**Git Commit**: HEAD (uncommitted)
**Branch**: master
**Repository**: poke

## Research Question

Analyze the most advanced way of battling in this codebase and determine if there are ways to improve.

## Summary

The codebase implements a sophisticated Pokemon battle AI training framework with **four tiers of agents**: random baselines, heuristic agents, behavior cloning (BC) from human replays, and offline reinforcement learning (IQL/CQL). The **most advanced approach is Offline RL with IQL/CQL algorithms** combined with MLP or Transformer policy networks, trained on human replay data.

Compared to state-of-the-art research (Metamon, PokéChamp, VGC-Bench published in 2024-2025), there are **significant improvement opportunities** in architecture, training methodology, and features that could potentially boost performance from current levels to human-competitive (top 10-30% on Pokemon Showdown ladder).

---

## Detailed Findings

### 1. Current Most Advanced Approach: Offline RL

#### Implementation Overview

The most advanced battling approach in this codebase is **Offline Reinforcement Learning** using either:
- **IQL (Implicit Q-Learning)**: `src/poke/training/iql.py`
- **CQL (Conservative Q-Learning)**: `src/poke/training/cql.py`

Both algorithms train on pre-collected trajectory data from Pokemon Showdown replays without online environment interaction.

#### Architecture Stack

```
Human Replays (JSONL)
    ↓
TrajectoryDataset (dataset.py)
    ↓
FeaturePreprocessor (preprocessing.py)
    ↓
StateEncoder (state_encoder.py) → 256-dim state
    ↓
MLPPolicy / TransformerPolicy
    ↓
ActionMask → 10 discrete actions (4 moves + 6 switches)
    ↓
OfflineRLTrainer (rl_trainer.py)
```

#### Key Technical Details

**State Representation** (`state_encoder.py:38-43`):
- Team ID embedding: 32 dims
- Weather embedding: 16 dims
- Terrain embedding: 16 dims
- Pokemon features: 384 dims (12 Pokemon × 32 features)
- Hazards: 16 dims
- **Total input**: 464 dims → encoded to 256 dims

**Policy Networks**:
- **MLPPolicy** (`policy.py`): 3 ResidualBlocks, ~400K parameters
- **TransformerPolicy** (`transformer_policy.py`): 2-layer encoder, ~1.66M parameters

**Offline RL Algorithms**:
- **IQL**: Uses expectile regression (τ=0.7) + advantage-weighted BC
- **CQL**: Adds conservative penalty (α=5.0) to prevent OOD action overestimation

---

### 2. Comparison with State-of-the-Art

| Aspect | Current Codebase | Metamon (SOTA) | PokéChamp (SOTA) |
|--------|------------------|----------------|------------------|
| **Model Size** | ~1.66M params | 200M params | GPT-4o / Llama 8B |
| **Architecture** | 2-layer Transformer | Decoder-only Transformer | Minimax + LLM |
| **Training Data** | Unknown scale | 1M human + 4M self-play | 2M+ battles |
| **RL Algorithm** | IQL/CQL | Custom offline RL | Prompting |
| **Performance** | Untested on ladder | Top 10% players | 1268-1500 Elo |
| **Self-Play** | Not implemented | Yes (fine-tuning) | No |

---

### 3. Identified Improvement Opportunities

#### A. Model Architecture Improvements

**1. Scale Up Model Size**
- Current: ~400K (MLP) to 1.66M (Transformer) parameters
- SOTA: 15M to 200M parameters
- **Recommendation**: Increase hidden dimensions from 256 to 512-1024, add more layers (6-12)

**2. Use Decoder-Only Transformer Architecture**
- Current: Encoder-only transformer processes fixed sequence
- SOTA (Metamon): Decoder-only transformer with causal attention
- **Benefit**: Better for sequential decision-making, supports autoregressive generation

**3. Add Value Classification**
- Current: Scalar value head output
- SOTA (Metamon): Classifies value into bins instead of regression
- **Benefit**: More stable training, better uncertainty handling

**4. Improve Pokemon Encoding**
- Current: Simple 32-dim feature vector per Pokemon
- **Recommended additions**:
  - Species embeddings (like Metamon's learned embeddings)
  - Move embeddings for known movesets
  - Ability and item embeddings
  - Per-Pokemon attention aggregation

#### B. Training Methodology Improvements

**1. Add Self-Play Fine-Tuning Stage**
```
Current:  BC → Offline RL → Deploy
SOTA:     BC → Offline RL → Self-Play → Deploy
```
- Self-play allows policy to improve beyond human demonstration data
- Metamon uses ~4M self-play battles after 1M human battles

**2. Implement Fictitious Play or PSRO**
- VGC-Bench found **Fictitious Play** most reliable for Pokemon
- Maintains distribution over past policies to train against
- Prevents catastrophic forgetting and non-transitive cycling

**3. Train on Higher Quality Data**
- Filter for high-Elo games (>1600 rating)
- PokéChamp uses 500K high-Elo games from 2M+ total
- Quality > quantity for offline RL

**4. Implement Curriculum Learning**
- Start with simpler opponents (random, heuristic)
- Gradually increase difficulty
- Shown to improve sample efficiency in game-playing AI

#### C. Feature Engineering Improvements

**1. Better State Representation**
Currently missing:
- Move PP tracking (out of PP = can't use move)
- Boosts/stat changes for opponent Pokemon
- Turn count and game phase information
- Previous actions (for opponent modeling)
- Tera type and Terastallization status (Gen 9)
- Dynamax status and turns remaining (Gen 8)

**2. Add History/Context Window**
- Current: Single-turn observations only
- SOTA: Full battle history for opponent adaptation
- TransformerPolicy supports this but requires proper trajectory formatting

**3. Implement Opponent Modeling**
- Track opponent's revealed Pokemon, moves, items
- Estimate opponent's team composition from partial information
- Critical for imperfect information game

#### D. Search and Planning Improvements

**1. Add MCTS with Neural Network Guidance**
- MIT thesis showed MCTS + neural network hybrid effective
- Use policy network as prior, value network for evaluation
- Challenge: 20-second time limit on Pokemon Showdown

**2. Implement Minimax Reasoning**
- PokéChamp's approach: Consider opponent's best responses
- Can be approximated with learned world model
- Suitable for double-battles with simultaneous moves

**3. Add Monte Carlo Rollouts**
- Simulate forward using learned policy for value estimation
- Improve action selection at inference time

#### E. Infrastructure Improvements

**1. Larger Batch Sizes**
- Current: 256
- Recommendation: 1024-4096 with gradient accumulation
- Better for Transformer training stability

**2. Mixed Precision Training**
- Use fp16/bf16 for faster training on larger models
- Essential for scaling to 200M+ parameters

**3. Better Logging and Evaluation**
- Implement ELO rating against baseline agents
- Track win rates vs specific opponent types
- Add replay saving for debugging

---

### 4. Prioritized Improvement Roadmap

**Phase 1: Quick Wins (1-2 weeks)**
1. Add more state features (PP, boosts, turn count)
2. Filter training data for high-Elo games
3. Increase model hidden dimension to 512
4. Add proper evaluation harness with win rates

**Phase 2: Architecture Upgrade (2-4 weeks)**
1. Implement decoder-only Transformer (like Metamon)
2. Add learned embeddings for species/moves/items
3. Scale to 50M+ parameters
4. Implement value classification head

**Phase 3: Training Improvements (4-8 weeks)**
1. Add self-play fine-tuning stage
2. Implement Fictitious Play training
3. Collect/curate larger high-quality dataset
4. Add opponent modeling features

**Phase 4: Advanced Features (8+ weeks)**
1. Implement MCTS with neural network guidance
2. Add Monte Carlo rollout value estimation
3. Train generation-specific models
4. Deploy and test on Pokemon Showdown ladder

---

## Code References

### Current Implementation Files

- `src/poke/training/iql.py` - IQL algorithm implementation
- `src/poke/training/cql.py` - CQL algorithm implementation
- `src/poke/training/rl_trainer.py` - Offline RL training loop
- `src/poke/training/bc_trainer.py` - Behavior cloning trainer
- `src/poke/models/policy.py` - MLP policy network
- `src/poke/models/transformer_policy.py` - Transformer policy network
- `src/poke/models/state_encoder.py` - Battle state encoding
- `src/poke/agents/nn_agent.py` - Neural network agent for battles
- `src/poke/agents/heuristic_agent.py` - Heuristic baselines

### External References

**State-of-the-Art Papers:**
- [Metamon (2025)](https://arxiv.org/html/2504.04395v1) - 200M param transformer, top 10% players
- [PokéChamp (2025)](https://arxiv.org/html/2503.04094v1) - Minimax LLM agent, 76% vs PokéLLMon
- [VGC-Bench (2025)](https://arxiv.org/html/2506.10326v2) - Multi-agent benchmark, beat VGC pro

**Open Source Projects:**
- [Metamon GitHub](https://github.com/UT-Austin-RPL/metamon)
- [poke-env](https://github.com/hsahovic/poke-env) - Standard Pokemon battle interface
- [VGC-Bench](https://github.com/cameronangliss/VGC-Bench)

**Datasets:**
- [Metamon Parsed Replays](https://huggingface.co/datasets/jakegrigsby/metamon-parsed-replays) - 1M battles
- [Pokemon Showdown Replays](https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays) - 2005-2025

---

## Architecture Documentation

### Current Training Pipeline

```
Raw Battle Replays (Pokemon Showdown)
    ↓
scraper.py → Fetch replays via API
    ↓
parser.py → Parse logs to ParsedBattle
    ↓
converter.py → Convert to Trajectory format
    ↓
trajectories.jsonl (training data)
    ↓
┌─────────────────────────────────────────┐
│ Option A: Behavior Cloning              │
│   BCTrainer → Cross-entropy loss        │
│   Output: BC checkpoint                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ Option B: Offline RL                    │
│   IQL: expectile + advantage weighting  │
│   CQL: conservative penalty             │
│   Can warm-start from BC checkpoint     │
│   Output: RL checkpoint                 │
└─────────────────────────────────────────┘
    ↓
NeuralNetworkAgent → Battle deployment
```

### Action Space Design

| Index | Action Type | Description |
|-------|-------------|-------------|
| 0 | Move | Use move in slot 1 |
| 1 | Move | Use move in slot 2 |
| 2 | Move | Use move in slot 3 |
| 3 | Move | Use move in slot 4 |
| 4 | Switch | Switch to Pokemon in slot 1 |
| 5 | Switch | Switch to Pokemon in slot 2 |
| 6 | Switch | Switch to Pokemon in slot 3 |
| 7 | Switch | Switch to Pokemon in slot 4 |
| 8 | Switch | Switch to Pokemon in slot 5 |
| 9 | Switch | Switch to Pokemon in slot 6 |

---

## Related Research

- This is the first research document for this codebase

---

## Open Questions

1. **What is the current performance** of trained agents on Pokemon Showdown ladder?
2. **How much training data** has been collected? (Metamon used 1M+ human games)
3. **Which generation/format** is the primary target? (Gen 9 OU, Random Battles, etc.)
4. **Are there computational constraints** that limit model scaling?
5. **Is online self-play** feasible with the current poke-env integration?

---

## Conclusions

The codebase has a solid foundation with offline RL (IQL/CQL), but there is a significant gap compared to state-of-the-art Pokemon AI that has achieved human-level performance. The most impactful improvements would be:

1. **Scale model size** from ~1M to 50-200M parameters
2. **Add self-play fine-tuning** after offline RL
3. **Improve state representation** with better Pokemon/move embeddings
4. **Collect/filter high-quality data** (high-Elo games)

With these improvements, based on published results, it's realistic to target **top 10-30% performance on Pokemon Showdown ladder** (Elo 1300-1500).
