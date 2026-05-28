"""RELIEF reward computation and GAE advantage estimation.

Copied verbatim from the original RELIEF repository to preserve behaviour.
"""

from __future__ import annotations

import numpy as np
import torch


class RunningMeanStd:
    """Dynamically calculate mean and std."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        return (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        return x / (self.running_ms.std + 1e-8)

    def reset(self):
        self.R = np.zeros(self.shape)


def reward_reshape(rewards, reward_clip=0.1):
    """Reshape episode rewards via loss decrease."""
    return (rewards[:-1] - rewards[1:]).clip(-reward_clip, reward_clip)


def reshape_reward(rewards, valid_steps, reward_clip=0.1):
    use_steps = valid_steps + 1
    episode_rewards = []
    for i in range(rewards.size(0)):
        valid_rewards = rewards[i][: use_steps[i]]
        decrease = valid_rewards[:-1] - valid_rewards[1:]
        episode_reward = torch.clamp(decrease, min=-reward_clip, max=reward_clip)
        episode_rewards.append(episode_reward.cpu().numpy())
    return episode_rewards


def reward_scaling(rewards, reward_scaler):
    scaled_rewards = []
    for reward in rewards:
        scaled_rewards.append(reward_scaler(reward))
    scaled_rewards = np.stack(scaled_rewards, axis=0)
    reward_scaler.reset()
    return torch.from_numpy(scaled_rewards)


def compute_adv_ret(args, critic, states, rewards, next_states, dones, nodes_per_graph, reward_transform):
    """Calculate advantage and return-to-go of an episode (GAE)."""
    device = states[0].device
    num_episode = len(rewards)

    if reward_transform.running_ms.n == 0:
        for i in range(num_episode):
            for reward in rewards[i]:
                reward_transform.running_ms.update(reward)

    scaled_rewards = []
    for i in range(num_episode):
        step_scaled_rewards = []
        for reward in rewards[i]:
            step_scaled_rewards.append(reward_transform(reward))
        scaled_rewards.append(torch.tensor(np.stack(step_scaled_rewards)).squeeze(-1).to(device))

    states = torch.split(states, nodes_per_graph)
    next_states = torch.split(next_states, nodes_per_graph)
    dones = torch.split(dones, nodes_per_graph)

    advs, rets = [], []
    with torch.no_grad():
        for i in range(num_episode):
            state_values = critic(states[i])
            next_state_values = critic(next_states[i])
            gae = torch.tensor(0).to(device)
            adv = []
            deltas = scaled_rewards[i] + args.gamma * (1.0 - dones[i]) * next_state_values - state_values
            for delta, done in zip(reversed(deltas), reversed(dones[i])):
                gae = delta + args.gamma * args.lam * gae * (1.0 - done)
                adv.insert(0, gae)
            adv = torch.stack(adv)
            ret = adv + state_values
            advs.append(adv)
            rets.append(ret)

    return torch.cat(advs), torch.cat(rets), torch.stack([ret[0] for ret in rets]), scaled_rewards
