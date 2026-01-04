# PRD.md
## Project: Pokémon Showdown Gen9 OU AI (Offline RL)

---

## 1. Overview

**Goal**  
Build an AI agent capable of playing **Pokémon Showdown – Gen9 OU** at a competitive level using **offline reinforcement learning** trained on historical battle datasets.

The agent will focus exclusively on **in-battle decision making** (moves and switches), trained from real human games.

**Non-Goals**
- Team building optimization or learning
- Live ladder deployment at scale
- Exploiting Showdown mechanics
- Full-information solving (imperfect information is accepted)

---

## 2. Target Format & Constraints

- **Battle Format:** Gen9 OU
- **Game Type:** Turn-based, imperfect information
- **Actions per Turn:**
  - Select a legal move
  - Switch to a legal Pokémon
- **Victory Condition:** Standard Showdown win/loss rules

---

## 3. Team Selection Strategy (Locked Decision)

### Approach
- Use a **static pool of externally sourced Gen9 OU teams**
- Teams are selected **before battle start**
- One team is sampled uniformly at random per battle
- Team composition remains fixed during training and evaluation

### Team Sources
- **Primary:** Smogon Gen9 OU Sample Teams
- **Secondary (optional):** High-ladder OU teams (Top 500 Elo)

### Rationale
- Removes team strength as a confounding variable
- Enables clean attribution of performance to decision quality
- Improves reproducibility and debugging
- Keeps RL scope focused and tractable

### Representation
- Each team assigned a stable `TeamID`
- `TeamID` is included in the agent observation
- Teams are versioned and immutable (e.g. `teams/gen9ou/v1/`)

### Explicitly Out of Scope
- Team evolution or optimization
- Joint learning of team selection and battle policy
- Meta-adaptive team generation

---

## 4. High-Level Architecture

```
OU Replay Dataset
        ↓
Data Parser (state/action trajectories)
        ↓
Supervised Pretraining (Behavior Cloning)
        ↓
Offline RL Fine-Tuning
        ↓
Battle Simulation (poke-env)
        ↓
Evaluation Harness
```

---

## 5. Data Requirements

### Dataset
- Pokémon Showdown replay data filtered to **Gen9 OU**
- Minimum size: **100k battles**
- Preferred size: **500k–1M battles**

### Parsed Trajectory Schema

Each battle is converted into a sequence of:

```
State_t:
- turn_number
- team_id
- active_pokemon
- own_team_state (HP, status, fainted)
- known_opponent_state
- field_conditions
- hazards
- weather

Action_t:
- action_type (MOVE | SWITCH)
- target_index

Reward_t:
- sparse terminal reward (+1 win / -1 loss)
- optional shaping (damage, KO events)
```

---

## 6. Model Requirements

### Model Type
- Neural network policy trained via:
  - Supervised learning (behavior cloning)
  - Offline reinforcement learning

### Candidate Architectures
- MLP with engineered features
- Transformer over turn sequences
- Actor-Critic variants (CQL, IQL)

### Output
- Probability distribution over **legal actions**
- Invalid actions must be masked

---

## 7. Training Strategy

### Phase 1: Supervised Pretraining
- Train on human action data
- Objective: maximize likelihood of demonstrated actions
- Purpose: stabilize policy initialization

### Phase 2: Offline RL Fine-Tuning
- Train using logged rewards
- Conservative learning to avoid distribution shift
- Algorithms:
  - Conservative Q-Learning (CQL)
  - Implicit Q-Learning (IQL)

---

## 8. Environment & Simulation

- **Environment Library:** poke-env
- **Server:** Local Pokémon Showdown instance
- **Battle Modes:**
  - Agent vs baseline bots
  - Agent vs frozen policy snapshots
- **Format Lock:** Gen9 OU only

---

## 9. Evaluation Metrics

### Primary Metrics
- Winrate vs baseline agents
- Simulated Elo
- Winrate by team archetype

### Secondary Metrics
- Illegal action rate
- Average game length
- Action entropy

### Regression
- Deterministic replay tests
- Fixed-seed evaluation battles

---

## 10. Milestones

### M1 – Dataset & Parsing
- Acquire Gen9 OU replay data
- Parse into trajectories
- Validate legality and consistency

### M2 – Team Pool
- Curate static team pool
- Implement team loader
- Add TeamID to observation

### M3 – Baseline Agents
- Random agent
- Heuristic (max-damage) agent

### M4 – Supervised Policy
- Feature encoder
- Action masking
- Training loop

### M5 – Offline RL
- Algorithm implementation
- Stabilization and checkpoints
- Hyperparameter tuning

### M6 – Evaluation Harness
- Automated battle runner
- Metrics collection
- Policy comparison tools

---

## 11. Risks & Mitigations

| Risk | Mitigation |
|----|----|
| Partial observability | Explicit unknown tokens |
| Dataset bias | Diverse Elo sampling |
| Illegal actions | Hard action masking |
| Overfitting | Cross-team evaluation |
| Training instability | Supervised warm start |

---

## 12. Tech Stack

- **Language:** Python
- **ML:** PyTorch
- **RL:** Stable-Baselines3 / custom
- **Environment:** poke-env
- **Data:** JSONL / HuggingFace Datasets
- **Tracking:** Weights & Biases (optional)

---

## 13. Success Criteria

- >60% winrate vs heuristic baseline
- Zero illegal actions
- Reproducible training runs
- Modular, PR-friendly architecture
