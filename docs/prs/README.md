# PR Execution Guide

This directory contains sharded PR documents for implementing the Pokémon Showdown Gen9 OU AI. Each PR is scoped for a junior engineer and specifies dependencies to enable parallel execution.

## Dependency Graph

```
PR-001 (Project Setup)
    ├── PR-002 (Showdown Server)
    │       └── PR-003 (poke-env Integration)
    │               ├── PR-010 (TeamID Observation) ←── PR-008 (Team Loader)
    │               ├── PR-011 (Random Agent)
    │               └── PR-012 (Heuristic Agent)
    │
    ├── PR-004 (Replay Scraper)
    │       └── PR-005 (Replay Parser)
    │               └── PR-006 (Trajectory Converter)
    │                       └── PR-007 (Data Validation)
    │
    ├── PR-008 (Team Loader)
    │       └── PR-009 (Team Curation)
    │
    └── PR-013 (State Encoder)
            └── PR-014 (Action Masking)
                    └── PR-015 (Policy Network)
                            └── PR-016 (Behavior Cloning)
                                    ├── PR-017 (BC Checkpointing)
                                    └── PR-018 (Offline RL Algorithms)
                                            └── PR-019 (Offline RL Training)
                                                    └── PR-020 (Evaluation Harness)
                                                            └── PR-021 (Metrics Collection)
```

## Execution Phases

### Phase 1: Foundation (Parallel)
| PR | Name | Dependencies |
|----|------|--------------|
| PR-001 | Project Setup | None |

### Phase 2: Infrastructure (Parallel after PR-001)
| PR | Name | Dependencies |
|----|------|--------------|
| PR-002 | Showdown Server | PR-001 |
| PR-004 | Replay Scraper | PR-001 |
| PR-008 | Team Loader | PR-001 |
| PR-013 | State Encoder | PR-001 |

### Phase 3: Integration (Parallel within groups)
| PR | Name | Dependencies |
|----|------|--------------|
| PR-003 | poke-env Integration | PR-001, PR-002 |
| PR-005 | Replay Parser | PR-001, PR-004 |
| PR-009 | Team Curation | PR-008 |
| PR-014 | Action Masking | PR-001, PR-013 |

### Phase 4: Core Features (Parallel within groups)
| PR | Name | Dependencies |
|----|------|--------------|
| PR-006 | Trajectory Converter | PR-005 |
| PR-010 | TeamID Observation | PR-003, PR-008 |
| PR-011 | Random Agent | PR-003, PR-010 |
| PR-012 | Heuristic Agent | PR-003, PR-010 |
| PR-015 | Policy Network | PR-013, PR-014 |

### Phase 5: Training (Sequential)
| PR | Name | Dependencies |
|----|------|--------------|
| PR-007 | Data Validation | PR-005, PR-006 |
| PR-016 | Behavior Cloning | PR-006, PR-013-015 |
| PR-017 | BC Checkpointing | PR-016 |

### Phase 6: RL & Evaluation (Sequential)
| PR | Name | Dependencies |
|----|------|--------------|
| PR-018 | Offline RL Algorithms | PR-015, PR-016 |
| PR-019 | Offline RL Training | PR-016, PR-018 |
| PR-020 | Evaluation Harness | PR-003, PR-011, PR-012, PR-015 |
| PR-021 | Metrics Collection | PR-020 |

## Parallel Execution Strategy

For maximum parallelism with a swarm:

**Wave 1 (1 engineer):**
- PR-001: Project Setup

**Wave 2 (4 engineers in parallel):**
- PR-002: Showdown Server
- PR-004: Replay Scraper
- PR-008: Team Loader
- PR-013: State Encoder

**Wave 3 (4 engineers in parallel):**
- PR-003: poke-env Integration
- PR-005: Replay Parser
- PR-009: Team Curation
- PR-014: Action Masking

**Wave 4 (5 engineers in parallel):**
- PR-006: Trajectory Converter
- PR-010: TeamID Observation
- PR-011: Random Agent
- PR-012: Heuristic Agent
- PR-015: Policy Network

**Wave 5 (2 engineers in parallel):**
- PR-007: Data Validation
- PR-016: Behavior Cloning

**Wave 6 (1 engineer):**
- PR-017: BC Checkpointing

**Wave 7 (2 engineers in parallel):**
- PR-018: Offline RL Algorithms
- PR-020: Evaluation Harness

**Wave 8 (2 engineers, sequential):**
- PR-019: Offline RL Training
- PR-021: Metrics Collection

## Tech Stack Summary

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Framework | PyTorch |
| Environment | poke-env |
| Data Format | JSONL, HuggingFace Datasets |
| Tracking | Weights & Biases (optional) |
| Testing | pytest |
| Linting | ruff, black, mypy |

## Success Criteria

From the PRD:
- >60% winrate vs heuristic baseline
- Zero illegal actions
- Reproducible training runs
- Modular, PR-friendly architecture

## Quick Start

```bash
# After all PRs merged:

# 1. Setup
pip install -e ".[dev]"
./scripts/setup_showdown.sh

# 2. Get data
python scripts/scrape_replays.py --max-replays 10000
python scripts/parse_replays.py data/raw/replays/gen9ou_replays.jsonl
python scripts/convert_trajectories.py data/raw/replays/gen9ou_replays.parsed.jsonl

# 3. Train BC
python scripts/train_bc.py --data data/processed/trajectories.jsonl --epochs 10

# 4. Train RL
python scripts/train_rl.py --data data/processed/trajectories.jsonl \
    --bc-checkpoint checkpoints/bc/best.pt --algorithm iql

# 5. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/rl/best.pt --battles 100
```
