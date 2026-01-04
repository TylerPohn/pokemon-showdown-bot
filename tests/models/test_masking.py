"""Tests for action masking."""
import pytest
import torch
import numpy as np
from unittest.mock import Mock

from poke.models.action_space import ActionSpace
from poke.models.masking import ActionMask, MaskedPolicy, ActionSelector

@pytest.fixture
def action_space():
    return ActionSpace()

@pytest.fixture
def mock_battle():
    battle = Mock()

    # 3 available moves
    battle.available_moves = [Mock(), Mock(), Mock()]

    # 2 available switches
    battle.available_switches = [Mock(), Mock()]

    return battle

def test_mask_shape(action_space, mock_battle):
    mask_gen = ActionMask(action_space)
    mask = mask_gen.get_mask(mock_battle)

    assert mask.shape == (10,)
    assert mask.dtype == bool

def test_mask_legal_moves(action_space, mock_battle):
    mask_gen = ActionMask(action_space)
    mask = mask_gen.get_mask(mock_battle)

    # First 3 moves should be legal
    assert mask[0] == True
    assert mask[1] == True
    assert mask[2] == True
    assert mask[3] == False  # No 4th move

    # First 2 switches should be legal (indices 4-5)
    assert mask[4] == True
    assert mask[5] == True
    assert mask[6] == False

def test_masked_policy_zeros_illegal():
    policy = MaskedPolicy(input_dim=64, action_dim=10)

    state = torch.randn(1, 64)
    mask = torch.tensor([[True, True, False, False, True, False, False, False, False, False]])

    probs = policy(state, mask)

    # Illegal actions should have ~0 probability
    assert probs[0, 2].item() < 1e-6
    assert probs[0, 3].item() < 1e-6

    # Legal actions should sum to ~1
    legal_sum = probs[0, mask[0]].sum().item()
    assert abs(legal_sum - 1.0) < 1e-6

def test_action_space_encoding():
    space = ActionSpace()

    # Move encoding
    assert space.encode_action("move", 0) == 0
    assert space.encode_action("move", 3) == 3

    # Switch encoding
    assert space.encode_action("switch", 0) == 4
    assert space.encode_action("switch", 5) == 9

def test_action_space_decoding():
    space = ActionSpace()

    assert space.decode_action(0) == ("move", 0)
    assert space.decode_action(3) == ("move", 3)
    assert space.decode_action(4) == ("switch", 0)
    assert space.decode_action(9) == ("switch", 5)
