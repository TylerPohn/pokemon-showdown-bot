# Plan: Try Trained BC Model

## Overview
The BC model has been trained on 1500+ rated players' replays and checkpoints exist at `models/bc/final.pt`. This plan outlines the integration work needed to run the model in battles.

## Current State
- **Trained Model**: `models/bc/final.pt` (8.4 MB, 10 epochs)
- **Agent Class**: `NeuralNetworkAgent` in `src/poke/agents/nn_agent.py`
- **Policy Architecture**: MLPPolicy with StateEncoder
- **Action Space**: 10 actions (4 moves + 6 switches)

---

## Integration Steps

### Step 1: Verify Model Loading
**Goal**: Ensure the checkpoint loads correctly with matching architecture

- [ ] Load `models/bc/final.pt` and inspect checkpoint structure
- [ ] Verify keys match expected: `model_state_dict`, `optimizer_state_dict`, `epoch`
- [ ] Check tensor shapes against `EncoderConfig` defaults
- [ ] Ensure device handling (CPU/GPU) works

### Step 2: Verify Observation Pipeline
**Goal**: Ensure observations from live battles match training data format

- [ ] Compare `ObservationBuilder` output format with training trajectory format
- [ ] Check feature dimensions match model's expected input size
- [ ] Verify team_id encoding matches training
- [ ] Test preprocessing pipeline with sample battle state

### Step 3: Verify Action Masking
**Goal**: Ensure legal move filtering works correctly

- [ ] Test `ActionMask` generation from battle state
- [ ] Verify mask application to policy logits
- [ ] Test `ActionSelector` conversion from action indices to battle orders
- [ ] Handle edge cases (forced switches, fainted Pokemon)

### Step 4: Create Integration Test Script
**Goal**: Quick smoke test for the full pipeline

```python
# scripts/test_bc_agent.py
# - Load model checkpoint
# - Create NeuralNetworkAgent
# - Run 1 battle vs RandomAgent
# - Print action choices for debugging
```

### Step 5: Run Evaluation Battles
**Goal**: Measure model performance against baselines

- [ ] Battle vs PureRandomAgent (should win ~90%+)
- [ ] Battle vs MaxDamageAgent (competitive baseline)
- [ ] Battle vs SmartHeuristicAgent (stronger baseline)
- [ ] Run 100+ battles per matchup for statistical significance

### Step 6: Debug & Fix Issues
**Goal**: Address any integration problems discovered

Common issues to watch for:
- [ ] Feature normalization mismatches
- [ ] Species/move ID mapping differences
- [ ] Observation dimension mismatches
- [ ] Action masking edge cases
- [ ] Team pool loading issues

---

## Implementation Order

1. **Quick Smoke Test** (30 min)
   - Write minimal script to load model and run 1 battle
   - Identify any immediate errors

2. **Fix Observation Format** (if needed)
   - Compare training data format vs live observation format
   - Add any missing feature preprocessing

3. **Fix Action Conversion** (if needed)
   - Ensure action indices map to correct battle orders
   - Handle mega/z-move/dynamax/terastallize if applicable

4. **Full Evaluation Run**
   - Run tournament against all baselines
   - Generate performance report

---

## Success Criteria

- [ ] BC agent loads without errors
- [ ] BC agent completes full battles without crashes
- [ ] BC agent beats PureRandomAgent >80% of the time
- [ ] BC agent is competitive with MaxDamageAgent (>40% winrate)

---

## Files to Modify/Create

| File | Action | Purpose |
|------|--------|---------|
| `scripts/test_bc_agent.py` | Create | Quick integration test |
| `src/poke/agents/nn_agent.py` | May modify | Fix any loading issues |
| `src/poke/agents/observation.py` | May modify | Match training format |
| `src/poke/models/preprocessing.py` | May modify | Feature preprocessing |

---

## Commands

```bash
# Step 1: Verify checkpoint
python -c "import torch; ckpt = torch.load('models/bc/final.pt', map_location='cpu'); print(ckpt.keys())"

# Step 4: Run smoke test
python scripts/test_bc_agent.py

# Step 5: Run full evaluation
python scripts/evaluate.py --agent bc --checkpoint models/bc/final.pt --battles 100
```
