# modified from stable baselines implementation
# assume the wrappers already normalized the observation
import random
from typing import Optional, List, Union, Dict

import numpy as np
from numpy import ndarray
from copy import deepcopy
from .segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer(object):
    def __init__(self, size: int, noflush_size: int = 0):
        """
        Implements a ring buffer (FIFO).

        :param size: (int)  Max number of transitions to store in the buffer. When the buffer overflows the old
            memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._noflush_size = noflush_size
        assert noflush_size <= size
        self._next_idx = 0

    def __len__(self) -> int:
        return len(self._storage)

    @property
    def storage(self):
        """[(Union[np.ndarray, int], Union[np.ndarray, int], float, Union[np.ndarray, int], bool)]: content of the replay buffer"""
        return self._storage

    @property
    def buffer_size(self) -> int:
        """float: Max capacity of the buffer"""
        return self._maxsize

    def can_sample(self, n_samples: int) -> bool:
        """
        Check if n_samples samples can be sampled
        from the buffer.

        :param n_samples: (int)
        :return: (bool)
        """
        return len(self) >= n_samples

    def is_full(self) -> int:
        """
        Check whether the replay buffer is full or not.

        :return: (bool)
        """
        return len(self) == self.buffer_size

    def add(self, obs_t, action, reward, obs_tp1, done, is_pad):
        """
        add a new transition to the buffer

        :param obs_t: (Union[np.ndarray, int, Dict[str, np.ndarray]]) the last observation
        :param action: (Union[np.ndarray, int]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Union[np.ndarray, int, Dict[str, np.ndarray]]) the current observation
        :param done: (bool) is the episode done
        """
        data = (obs_t, action, reward, obs_tp1, done, is_pad)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        if self._next_idx < self._noflush_size:
            self._next_idx = (self._next_idx + 1) % self._maxsize
        else:
            self._next_idx = self._noflush_size + (
                self._next_idx + 1 - self._noflush_size
            ) % (self._maxsize - self._noflush_size)

    def extend(self, obs_t, action, reward, obs_tp1, done):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
                but expects iterables and arrays with more than 1 dimensions
        """
        for data in zip(obs_t, action, reward, obs_tp1, done):
            if self._next_idx >= len(self._storage):
                self._storage.append(data)
            else:
                self._storage[self._next_idx] = data
            if self._next_idx < self._noflush_size:
                self._next_idx = (self._next_idx + 1) % self._maxsize
            else:
                self._next_idx = self._noflush_size + (
                    self._next_idx + 1 - self._noflush_size
                ) % (self._maxsize - self._noflush_size)

    # @staticmethod
    # def _normalize_obs(obs: np.ndarray,
    #                    env: Optional[VecNormalize] = None) -> np.ndarray:
    #     """
    #     Helper for normalizing the observation.
    #     """
    #     if env is not None:
    #         return env.normalize_obs(obs)
    #     return obs

    # @staticmethod
    # def _normalize_reward(reward: np.ndarray,
    #                       env: Optional[VecNormalize] = None) -> np.ndarray:
    #     """
    #     Helper for normalizing the reward.
    #     """
    #     if env is not None:
    #         return env.normalize_reward(reward)
    #     return reward

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return (
            np.array(obses_t),
            np.array(actions),
            np.array(rewards),
            np.array(obses_tp1),
            np.array(dones),
        )

    def sample(self, batch_size: int, **_kwargs):
        """
        Sample a batch of experiences.

        :param batch_size: (int) How many transitions to sample.
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)


class DictObsReplayBuffer(ReplayBuffer):
    def __init__(
        self, size: int, image_keys=("third-rgb", "wrist-rgb"), noflush_size: int = 0
    ):
        super().__init__(size, noflush_size)
        self.image_keys = image_keys
        self._obs_keys_and_types: Dict = None

    def add(
        self,
        obs_t: Dict[str, np.ndarray],
        action,
        reward,
        obs_tp1: Dict[str, np.ndarray],
        done,
        is_pad,
    ):
        if self._obs_keys_and_types is None:
            self._obs_keys_and_types = {}
            for key in obs_t.keys():
                self._obs_keys_and_types[key] = type(obs_t[key])
        for key in obs_t.keys():
            if key in self.image_keys:
                obs_t[key] = obs_t[key].astype(np.uint8)
                obs_tp1[key] = obs_tp1[key].astype(np.uint8)
        return super().add(obs_t.copy(), action, reward, obs_tp1.copy(), done, is_pad)

    def extend(
        self,
        obs_t: Dict[str, np.ndarray],
        action,
        reward,
        obs_tp1: Dict[str, np.ndarray],
        done,
        is_pad,
    ):
        bsz = reward.shape[0]
        for i in range(bsz):
            self.add(
                {key: obs_t[key][i] for key in obs_t.keys()},
                action[i],
                reward[i],
                {key: obs_tp1[key][i] for key in obs_tp1.keys()},
                done[i],
                is_pad[i],
            )

    def _encode_sample(self, idxes: Union[List[int], np.ndarray]):
        obses_t = {key: [] for key in self._obs_keys_and_types}
        actions = []
        rewards = []
        obses_tp1 = {key: [] for key in self._obs_keys_and_types}
        dones = []
        is_pads = []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done, is_pad = data
            for key in self._obs_keys_and_types:
                obses_t[key].append(obs_t[key])
                obses_tp1[key].append(obs_tp1[key])
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            is_pads.append(is_pad)
        for key in self._obs_keys_and_types:
            if self._obs_keys_and_types[key] == np.ndarray:
                obses_t[key] = np.array(obses_t[key])
                obses_tp1[key] = np.array(obses_tp1[key])
        return (
            obses_t,
            np.array(actions),
            np.array(rewards),
            obses_tp1,
            np.array(dones),
            np.array(is_pads),
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, size, alpha, noflush_size: int = 0):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(PrioritizedReplayBuffer, self).__init__(size, noflush_size)
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done, is_pad):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, is_pad)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done, is_pad):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done, is_pad)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority**self._alpha
            self._it_min[idx] = self._max_priority**self._alpha
            if idx < self._noflush_size:
                idx = (idx + 1) % self._maxsize
            else:
                idx = self._noflush_size + (idx + 1 - self._noflush_size) % (
                    self._maxsize - self._noflush_size
                )

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities**self._alpha
        self._it_min[idxes] = priorities**self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))


class DictObsPrioritizedReplayBuffer(DictObsReplayBuffer):
    # The same as PrioritizedReplayBuffer
    def __init__(self, size, alpha, noflush_size: int = 0):
        """
        Create Prioritized Replay buffer.

        See Also ReplayBuffer.__init__

        :param size: (int) Max number of transitions to store in the buffer. When the buffer overflows the old memories
            are dropped.
        :param alpha: (float) how much prioritization is used (0 - no prioritization, 1 - full prioritization)
        """
        super(DictObsPrioritizedReplayBuffer, self).__init__(
            size, noflush_size=noflush_size
        )
        assert alpha >= 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, obs_t, action, reward, obs_tp1, done, is_pad):
        """
        add a new transition to the buffer

        :param obs_t: (Any) the last observation
        :param action: ([float]) the action
        :param reward: (float) the reward of the transition
        :param obs_tp1: (Any) the current observation
        :param done: (bool) is the episode done
        """
        idx = self._next_idx
        super().add(obs_t, action, reward, obs_tp1, done, is_pad)
        self._it_sum[idx] = self._max_priority**self._alpha
        self._it_min[idx] = self._max_priority**self._alpha

    def extend(self, obs_t, action, reward, obs_tp1, done, is_pad):
        """
        add a new batch of transitions to the buffer

        :param obs_t: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the last batch of observations
        :param action: (Union[Tuple[Union[np.ndarray, int]]], np.ndarray]) the batch of actions
        :param reward: (Union[Tuple[float], np.ndarray]) the batch of the rewards of the transition
        :param obs_tp1: (Union[Tuple[Union[np.ndarray, int]], np.ndarray]) the current batch of observations
        :param done: (Union[Tuple[bool], np.ndarray]) terminal status of the batch

        Note: uses the same names as .add to keep compatibility with named argument passing
            but expects iterables and arrays with more than 1 dimensions
        """
        idx = self._next_idx
        super().extend(obs_t, action, reward, obs_tp1, done, is_pad)
        while idx != self._next_idx:
            self._it_sum[idx] = self._max_priority**self._alpha
            self._it_min[idx] = self._max_priority**self._alpha
            if idx < self._noflush_size:
                idx = (idx + 1) % self._maxsize
            else:
                idx = self._noflush_size + (idx + 1 - self._noflush_size) % (
                    self._maxsize - self._noflush_size
                )

    def _sample_proportional(self, batch_size):
        mass = []
        total = self._it_sum.sum(0, len(self._storage) - 1)
        # TODO(szymon): should we ensure no repeats?
        mass = np.random.random(size=batch_size) * total
        idx = self._it_sum.find_prefixsum_idx(mass)
        return idx

    def sample(self, batch_size: int, beta: float = 0):
        """
        Sample a batch of experiences.

        compared to ReplayBuffer.sample
        it also returns importance weights and idxes
        of sampled experiences.

        :param batch_size: (int) How many transitions to sample.
        :param beta: (float) To what degree to use importance weights (0 - no corrections, 1 - full correction)
        :param env: (Optional[VecNormalize]) associated gym VecEnv
            to normalize the observations/rewards when sampling
        :return:
            - obs_batch: (np.ndarray) batch of observations
            - act_batch: (numpy float) batch of actions executed given obs_batch
            - rew_batch: (numpy float) rewards received as results of executing act_batch
            - next_obs_batch: (np.ndarray) next set of observations seen after executing act_batch
            - done_mask: (numpy bool) done_mask[i] = 1 if executing act_batch[i] resulted in the end of an episode
                and 0 otherwise.
            - weights: (numpy float) Array of shape (batch_size,) and dtype np.float32 denoting importance weight of
                each sampled transition
            - idxes: (numpy int) Array of shape (batch_size,) and dtype np.int32 idexes in buffer of sampled experiences
        """
        assert beta > 0

        idxes = self._sample_proportional(batch_size)
        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self._storage)) ** (-beta)
        p_sample = self._it_sum[idxes] / self._it_sum.sum()
        weights = (p_sample * len(self._storage)) ** (-beta) / max_weight
        encoded_sample = self._encode_sample(idxes)
        return tuple(list(encoded_sample) + [weights, idxes])

    def update_priorities(self, idxes, priorities):
        """
        Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer
        to priorities[i].

        :param idxes: ([int]) List of idxes of sampled transitions
        :param priorities: ([float]) List of updated priorities corresponding to transitions at the sampled idxes
            denoted by variable `idxes`.
        """
        assert len(idxes) == len(priorities)
        assert np.min(priorities) > 0
        assert np.min(idxes) >= 0
        assert np.max(idxes) < len(self.storage)
        self._it_sum[idxes] = priorities**self._alpha
        self._it_min[idxes] = priorities**self._alpha

        self._max_priority = max(self._max_priority, np.max(priorities))
