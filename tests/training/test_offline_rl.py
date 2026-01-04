"""Tests for offline RL algorithms."""
import pytest
import torch

from poke.models.critic import QNetwork, TwinQNetwork, ValueNetwork
from poke.training.iql import IQL, IQLConfig

@pytest.fixture
def state_dim():
    return 64

@pytest.fixture
def action_dim():
    return 10

def test_q_network_output_shape(state_dim, action_dim):
    q_net = QNetwork(state_dim, action_dim)

    state = torch.randn(4, state_dim)
    action = torch.zeros(4, action_dim)
    action[:, 0] = 1  # One-hot

    q_value = q_net(state, action)

    assert q_value.shape == (4,)

def test_twin_q_min(state_dim, action_dim):
    twin_q = TwinQNetwork(state_dim, action_dim)

    state = torch.randn(4, state_dim)
    action = torch.zeros(4, action_dim)
    action[:, 0] = 1

    min_q = twin_q.min_q(state, action)

    assert min_q.shape == (4,)

def test_value_network(state_dim):
    v_net = ValueNetwork(state_dim)

    state = torch.randn(4, state_dim)
    value = v_net(state)

    assert value.shape == (4,)
