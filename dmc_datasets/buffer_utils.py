"""
Buffer utilities for handling replay memory and dataset loading in reinforcement learning.

This module provides classes and functions for managing experience replay buffers
and loading discretised DeepMind Control Suite datasets. The main components include
a ReplayBuffer class for storing and sampling transitions, and utilities for loading
and processing D4RL format datasets.
"""

import numpy as np
import os
import pickle
import torch
from typing import List, Tuple

from dmc_datasets.config import get_data_dir


class ReplayBuffer:
    """
    A replay buffer for storing and sampling transitions in reinforcement learning.

    This buffer supports standard replay operations as well as sequence sampling
    and loading from D4RL format datasets. It handles both regular transitions
    and empirical returns data. The buffer maintains transitions in a fixed-size
    circular buffer, automatically overwriting oldest data when full.

    Attributes:
        capacity: Maximum number of transitions that can be stored
        idx: Current insertion index in the buffer
        batch_size: Default batch size for sampling operations
        device: PyTorch device where tensors are stored
        states: Tensor of stored states
        actions: Tensor of stored actions
        rewards: Tensor of stored rewards
        next_states: Tensor of resulting states
        dones: Tensor of episode termination flags
        empirical_returns: Optional tensor of precomputed returns
        bootstrap_states: Optional tensor of bootstrap states
        bootstrap_dones: Optional tensor of bootstrap termination flags
        bootstrap_count: Optional tensor of bootstrap counts
        dataset_average_returns: Optional average return of loaded dataset
        done_flag: Optional tensor of episode completion flags
        state_dim: Dimension of state space
        action_dim: Dimension of action space
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, batch_size: int = 128, device: str = 'cpu') -> None:
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            batch_size: Default batch size for sampling
            device: PyTorch device for tensor storage
        """
        self.capacity = capacity
        self.idx = 0
        self.batch_size = batch_size
        self.device = device
        self.states = torch.zeros(size=(capacity, state_dim), dtype=torch.float).to(self.device)
        self.actions = torch.zeros(size=(capacity, action_dim), dtype=torch.long).to(self.device)
        self.rewards = torch.zeros(size=(capacity, 1), dtype=torch.float).to(self.device)
        self.next_states = torch.zeros(size=(capacity, state_dim), dtype=torch.float).to(self.device)
        self.dones = torch.zeros(size=(capacity, 1), dtype=torch.long).to(self.device)
        self.empirical_returns = None
        self.bootstrap_states = None
        self.bootstrap_dones = None
        self.bootstrap_count = None
        self.dataset_average_returns = None
        self.done_flag = None
        self.state_dim = state_dim
        self.action_dim = action_dim

    def push(self, state: torch.tensor, action: torch.tensor, reward: float, next_state: torch.tensor, done: bool) -> None:
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Resulting state
            done: Whether episode terminated
        """
        self.states[self.idx % self.capacity] = torch.from_numpy(state).to(self.device)
        self.actions[self.idx % self.capacity] = torch.from_numpy(action).to(self.device)
        self.rewards[self.idx % self.capacity] = reward
        self.next_states[self.idx % self.capacity] = torch.from_numpy(next_state).to(self.device)
        self.dones[self.idx % self.capacity] = int(done)
        self.idx += 1

    def sample(self, batch_size: int = None, idx: np.ndarray = None) -> Tuple[
        torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample a batch of transitions from the buffer.

        Args:
            batch_size: Number of transitions to sample (defaults to self.batch_size)
            idx: Optional specific indices to sample

        Returns:
            Tuple containing:
            - Batch of states
            - Batch of actions
            - Batch of rewards
            - Batch of next states
            - Batch of done flags
        """
        if not batch_size:
            batch_size = self.batch_size
        if idx is None:
            # When buffer large the probability of sampling a transition more than once -> 0
            idx = np.random.randint(low=0, high=min(self.idx, self.capacity), size=batch_size)
        return self.states[idx], self.actions[idx], self.rewards[idx], self.next_states[idx], self.dones[idx]

    def __len__(self) -> int:
        """Return current number of transitions in buffer."""
        return min(self.idx, self.capacity)

    def to_device(self, device: str = None) -> None:
        """
        Move buffer contents to specified device.

        Args:
            device: Target device (defaults to self.device)
        """
        if device is None:
            device = self.device
        self.states = self.states.to(device)
        self.actions = self.actions.to(device)
        self.rewards = self.rewards.to(device)
        self.next_states = self.next_states.to(device)
        self.dones = self.dones.to(device)

        if self.empirical_returns is not None:
            self.bootstrap_states = self.bootstrap_states.to(device)
            self.bootstrap_count = self.bootstrap_count.to(device)
            self.bootstrap_dones = self.bootstrap_dones.to(device)
            self.empirical_returns = self.empirical_returns.to(device)
            self.done_flag = self.done_flag.to(device)

    def reset_buffer(self, n_to_save: int = 0) -> None:
        """
        Reset buffer while optionally preserving some transitions.

        Args:
            n_to_save: Number of random transitions to preserve

        Raises:
            AssertionError: If n_to_save exceeds current buffer size
        """
        assert n_to_save < self.idx, "saving more than exists"
        idx = np.random.choice([i for i in range(self.idx)], replace=False, size=n_to_save)
        states = self.states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        next_states = self.next_states[idx]
        dones = self.dones[idx]
        self.states = torch.zeros(size=(self.capacity, self.state_dim), dtype=torch.float).to(self.device)
        self.actions = torch.zeros(size=(self.capacity, self.action_dim), dtype=torch.long).to(self.device)
        self.rewards = torch.zeros(size=(self.capacity, 1), dtype=torch.float).to(self.device)
        self.next_states = torch.zeros(size=(self.capacity, self.state_dim), dtype=torch.float).to(self.device)
        self.dones = torch.zeros(size=(self.capacity, 1), dtype=torch.long).to(self.device)
        self.states[:n_to_save] = states
        self.actions[:n_to_save] = actions
        self.rewards[:n_to_save] = rewards
        self.next_states[:n_to_save] = next_states
        self.dones[:n_to_save] = dones
        self.idx = n_to_save

    def load_from_d4rl(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
                       next_states: np.ndarray, dones: np.ndarray) -> None:
        """
        Load transitions from D4RL format arrays.

        Args:
            states: Array of states
            actions: Array of actions
            rewards: Array of rewards
            next_states: Array of next states
            dones: Array of done flags

        Raises:
            AssertionError: If data size exceeds buffer capacity
        """
        size = states.shape[0]
        assert size <= self.capacity, 'More data than buffer capacity!'
        self.states[:size] = torch.from_numpy(states).float().to(self.device)
        self.actions[:size] = torch.from_numpy(actions).float().to(self.device)
        self.rewards[:size] = torch.from_numpy(rewards).float().to(self.device)
        self.next_states[:size] = torch.from_numpy(next_states).float().to(self.device)
        self.dones[:size] = torch.from_numpy(dones).long().to(self.device)
        self.idx = size

    def sample_sequence(self, batch_size: int = None, rollout_len: int = None,
                        candidate_idx: int = None) -> Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
        """
        Sample sequences of transitions ensuring no episode boundaries are crossed.

        Args:
            batch_size: Number of sequences to sample
            rollout_len: Length of each sequence
            candidate_idx: Optional specific starting indices for sequences

        Returns:
            Tuple containing:
            - Batch of state sequences
            - Batch of action sequences
            - Batch of reward sequences
            - Batch of next state sequences
            - Batch of done flag sequences
        """
        if candidate_idx is None:
            candidate_idx = np.random.randint(low=0, high=min(self.idx, self.capacity) - rollout_len, size=batch_size)
        done_flag = torch.stack([self.done_flag[idx:idx + rollout_len] for idx in candidate_idx], dim=0)
        need_resample = (done_flag.sum(dim=-1) > 0).nonzero()
        for idx in need_resample:
            while True:
                new_candidate = np.random.randint(low=0, high=min(self.idx, self.capacity))
                if self.done_flag[new_candidate:new_candidate + rollout_len].sum() == 0:
                    candidate_idx[idx] = new_candidate
                    break
        return (torch.stack([self.states[candidate_idx + i] for i in range(rollout_len)], dim=1),
                torch.stack([self.actions[candidate_idx + i] for i in range(rollout_len)], dim=1),
                torch.stack([self.rewards[candidate_idx + i] for i in range(rollout_len)], dim=1),
                torch.stack([self.next_states[candidate_idx + i] for i in range(rollout_len)], dim=1),
                torch.stack([self.dones[candidate_idx + i] for i in range(rollout_len)], dim=1))


def get_buffer_from_d4rl(dataset: List, max_memory: int = None, batch_size: int = 256,
                         device: str = None) -> ReplayBuffer:
    """
    Create and initialize a replay buffer from D4RL format dataset.

    Args:
        dataset: List of transitions in D4RL format
        max_memory: Maximum buffer size (defaults to dataset size)
        batch_size: Batch size for sampling
        device: PyTorch device for tensor storage

    Returns:
        Initialized ReplayBuffer containing the dataset
    """
    states, actions, rewards, next_states, dones = get_from_list(dataset)
    if max_memory is None:
        max_memory = states.shape[0]
    buffer = ReplayBuffer(max_memory, states.shape[1], actions.shape[1], batch_size, device)
    buffer.load_from_d4rl(states, actions, rewards, next_states, dones)
    return buffer


def get_from_list(data: List) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert list of transitions to numpy arrays.

    Args:
        data: List of transition tuples

    Returns:
        Tuple of numpy arrays containing states, actions, rewards, next_states, and dones
    """
    states = [d[0] for d in data]
    actions = [d[1] for d in data]
    rewards = [d[2] for d in data]
    next_states = [d[3] for d in data]
    dones = [d[4] for d in data]
    return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)


def load_dataset(task_name: str = None, task: str = None,
                 level: str = None, bin_size: int = 3,
                 data_dir: str = None) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
    """
    Load a discretised DMC dataset from the specified directory.

    Args:
        task_name: Task name (e.g., 'walker')
        task: Specific task variant (e.g., 'walk')
        level: Difficulty level (e.g., 'expert')
        bin_size: Number of bins for action discretization
        data_dir: Optional override for data directory. If None, uses environment variable or default

    Returns:
        List of transition tuples (state, action, reward, next_state, done)

    Raises:
        FileNotFoundError: If dataset file cannot be found at specified location
    """
    if data_dir is None:
        data_dir = get_data_dir()

    filepath = os.path.join(data_dir, f'{bin_size}_bins', f'{task_name}-{task}-{level}')
    print(f"Loading dataset from: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Please download the datasets and set "
            "DMC_DISCRETE_DATA_DIR environment variable to point to their location."
        )

    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data
