"""
Environment utilities for DeepMind Control Suite discretization.

This module provides wrappers and utilities for discretizing continuous control environments
from the DeepMind Control Suite, supporting both atomic and factorized action spaces.
"""

import dm_control.suite as suite
import gymnasium as gym
from gymnasium import spaces
from itertools import product
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from dmc_datasets.buffer_utils import get_buffer_from_d4rl, load_dataset, ReplayBuffer
from dmc_datasets.constants import SCORES


def run_test(alg: Any, env: gym.Env, seed: int) -> float:
    """
    Run evaluation episode with the given algorithm and environment.

    Args:
        alg: Algorithm class with greedy_act method that takes state and returns action
            Must have method: greedy_act(state: np.ndarray) -> np.ndarray
        env: Gymnasium environment
        seed: Random seed for reproducibility

    Returns:
        Total episode reward
    """
    state, _ = env.reset(seed=seed)
    done = False
    score = 0
    while not done:
        action = alg.greedy_act(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        score += reward
    return score


class DMSuiteEnv(gym.Env):
    """
    Gymnasium wrapper for DeepMind Control Suite environments.

    Converts DM Control environments to the Gymnasium interface and handles
    flattening of observations.
    """

    def __init__(self, domain_name: str, task_name: str,
                 episode_len: Optional[int] = None, seed: Optional[int] = None) -> None:
        """
        Initialize DMC environment wrapper.

        Args:
            domain_name: DM Control domain (e.g., 'walker', 'cheetah')
            task_name: Specific task (e.g., 'walk', 'run')
            episode_len: Maximum episode length (defaults to env's limit)
            seed: Random seed for environment
        """
        if seed is not None:
            self.env = suite.load(domain_name, task_name, task_kwargs={'random': seed})
        else:
            self.env = suite.load(domain_name, task_name)
        num_actions = self.env.action_spec().shape[0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(num_actions,))

        time_step = self.env.reset()
        state_size = np.concatenate([v.flatten() for v in time_step.observation.values()]).shape[0]
        obs_high = np.array([np.inf for _ in range(state_size)], dtype=np.float32)
        obs_low = -obs_high
        self.observation_space = spaces.Box(obs_low, obs_high)
        self.episode_len = episode_len or self.env._step_limit
        self._time_step = None
        self.task_name = task_name
        self.domain_name = domain_name

    def reset(self, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Optional configurations

        Returns:
            Tuple of (initial observation, info dict)
        """
        if seed is None:
            self._time_step = self.env.reset()
        else:
            self.env = suite.load(self.domain_name, self.task_name, task_kwargs={'random': seed})
            self._time_step = self.env.reset()
        return np.concatenate([v.flatten() for v in self._time_step.observation.values()]), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take environment step.

        Args:
            action: Continuous action vector

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self._time_step = self.env.step(action)
        observation, reward, termination, info = (
            np.concatenate([v.flatten() for v in self._time_step.observation.values()]),
            self._time_step.reward,
            self._time_step.last(),
            {}
        )
        if self._time_step.last():
            info['truncated'] = not self._time_step.step_type.last()
        return observation, reward, False, self._time_step.last(), info


class AtomicDMEnv(gym.Wrapper):
    """
    Wrapper for atomic action space discretization.

    Converts continuous action space to single discrete action space
    by creating a lookup table of discretized actions.
    """

    def __init__(self, env: DMSuiteEnv, bin_size: int = 3) -> None:
        """
        Initialize atomic action wrapper.

        Args:
            env: DMSuiteWrapper environment
            bin_size: Number of bins per action dimension
        """
        super(AtomicDMEnv, self).__init__(env)
        lows = self.env.action_space.low
        highs = self.env.action_space.high
        self.action_lookups: Dict[int, List[float]] = {}
        bins = []
        for low, high in zip(lows, highs):
            bins.append(np.linspace(low, high, bin_size).tolist())
        for count, action in enumerate(product(*bins)):
            self.action_lookups[count] = list(action)
        self.num_actions = len(self.action_lookups)
        self.action_space = spaces.Discrete(self.num_actions)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take environment step with discrete action.

        Args:
            action: Discrete action index

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action = self.get_continuous_action(action)
        return super().step(action)

    def get_continuous_action(self, action: int) -> List[float]:
        """
        Convert discrete action to continuous action vector.

        Args:
            action: Discrete action index

        Returns:
            Continuous action vector
        """
        continuous_action = self.action_lookups[action]
        return continuous_action


class FactorisedDMEnv(gym.Wrapper):
    """
    Wrapper for factorized action space discretization.

    Converts continuous action space to multiple independent discrete action spaces,
    one for each action dimension.
    """

    def __init__(self, env: DMSuiteEnv, bin_size: Union[int, List[int], np.ndarray] = 3) -> None:
        """
        Initialize factorized action wrapper.

        Args:
            env: DMSuiteWrapper environment
            bin_size: Number of bins per action dimension, either single int or list
        """
        super(FactorisedDMEnv, self).__init__(env)
        self.num_subaction_spaces = self.env.action_space.shape[0]
        if isinstance(bin_size, int):
            self.bin_size = [bin_size] * self.num_subaction_spaces
        elif isinstance(bin_size, (list, np.ndarray)):
            assert len(bin_size) == self.num_subaction_spaces
            self.bin_size = bin_size
        lows = self.env.action_space.low
        highs = self.env.action_space.high
        self.action_lookups: Dict[int, Dict[int, float]] = {}
        for a, l, h in zip(range(self.num_subaction_spaces), lows, highs):
            self.action_lookups[a] = {}
            bins = np.linspace(l, h, self.bin_size[a])
            for count, b in enumerate(bins):
                self.action_lookups[a][count] = b
        self.action_space = spaces.MultiDiscrete(self.bin_size)

        scores = SCORES[f'{self.env.domain_name}-{self.env.task_name}'][f'bin_size_{bin_size}']
        self._random_score = scores['random']
        self._expert_score = scores['expert']

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Take environment step with factorized discrete action.

        Args:
            action: Vector of discrete actions, one per dimension

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        action = self.get_continuous_action(action)
        return super().step(action)

    def get_continuous_action(self, action: np.ndarray) -> List[float]:
        """
        Convert factorized discrete actions to continuous action vector.

        Args:
            action: Vector of discrete actions

        Returns:
            Continuous action vector
        """
        continuous_action = []
        for action_id, a in enumerate(action):
            continuous_action.append(self.action_lookups[action_id][a])
        return continuous_action

    def get_normalised_score(self, score: float) -> float:
        """
        Normalize score using random/expert performance baselines.

        Args:
            score: Raw environment score

        Returns:
            Normalized score between 0 (random) and 1 (expert)
        """
        return (score - self._random_score) / (self._expert_score - self._random_score)

    def load_dataset(self, level: str, data_dir: Optional[str] = None) -> ReplayBuffer:
        """
        Load dataset for current environment configuration.

        Args:
            level: Dataset difficulty level ('medium', 'expert', 'medium-expert', 'random-medium-expert')
            data_dir: Optional path to dataset directory

        Returns:
            Loaded replay buffer

        Raises:
            ValueError: If environment, bin size or level is not supported
        """
        domain_name = self.env.domain_name
        task_name = self.env.task_name
        bin_size = self.bin_size[0] if isinstance(self.bin_size, list) else self.bin_size

        # Check valid environments
        supported_envs = {
            'cheetah-run': [3],
            'finger-spin': [3],
            'fish-swim': [3],
            'quadruped-walk': [3],
            'humanoid-stand': [3],
            'dog-trot': [3, 10, 30, 50, 75, 100]
        }

        env_name = f"{domain_name}-{task_name}"
        if env_name not in supported_envs:
            raise ValueError(
                f"Environment {env_name} not supported. Supported environments are: "
                f"{', '.join(supported_envs.keys())}"
            )

        # Check valid bin size
        if bin_size not in supported_envs[env_name]:
            if env_name == 'dog-trot':
                raise ValueError(
                    f"Bin size {bin_size} not supported for {env_name}. "
                    f"Supported bin sizes are: {supported_envs[env_name]}"
                )
            else:
                raise ValueError(
                    f"Only bin size 3 is supported for {env_name}"
                )

        # Check valid difficulty level
        valid_levels = ['medium', 'expert', 'medium-expert', 'random-medium-expert']
        if level not in valid_levels:
            raise ValueError(
                f"Level {level} not supported. Supported levels are: {valid_levels}"
            )

        try:
            data = get_buffer_from_d4rl(load_dataset(domain_name, task_name, level, bin_size, data_dir))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset not found for {env_name} with bin_size={bin_size} and level={level}. "
                "Please ensure you have downloaded the datasets and set the correct data directory"
            )
        except Exception as e:
            raise Exception(
                f"Error loading dataset: {str(e)}. Please ensure the dataset format is correct "
                "and you have the necessary permissions to read the files"
            )

        return data


def make_env(task_name: str, task: str, bin_size: int = 3,
             factorised: bool = True) -> Union[FactorisedDMEnv, AtomicDMEnv]:
    """
    Create discretised DMC environment.

    Args:
        task_name: DM Control domain name
        task: Specific task name
        bin_size: Number of bins for discretisation
        factorised: Whether to use factorised (True) or atomic (False) discretisation

    Returns:
        Wrapped environment with discrete action space
    """
    if factorised:
        return FactorisedDMEnv(DMSuiteEnv(task_name, task), bin_size=bin_size)
    else:
        return AtomicDMEnv(DMSuiteEnv(task_name, task), bin_size=bin_size)
