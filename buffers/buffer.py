from typing import NamedTuple

import numpy as np
import jaxrl.numpy as jnp
import rlax


class BufferSamples(NamedTuple):
    observations: jnp.array
    actions: jnp.array
    values: jnp.array
    log_prob: jnp.array
    entropies: jnp.array
    advantages: jnp.array
    returns: jnp.array


class Buffer:
    def __init__(self, rollout_length, lam=0.99, gamma=0.9):
        self.rollout_length = rollout_length
        self.lam = lam
        self.gamma = gamma

        self.step = 0
        self.observations, self.actions, self.rewards, self.advantages = None, None, None, None
        self.returns, self.dones, self.values, self.log_probs = None, None, None, None

        self.reset()

    def reset(self):
        self.step = 0
        self.observations = np.zeros((self.rollout_length, self.obs_dim))
        self.actions = np.zeros((self.rollout_length, self.act_dim))
        self.rewards = np.zeros(self.rollout_length)
        self.returns = np.zeros(self.rollout_length)
        self.dones = np.zeros(self.rollout_length)
        self.values = np.zeros(self.rollout_length)
        self.log_probs = np.zeros(self.rollout_length)
        self.entropies = np.zeros(self.rollout_length)

    def add_exp(self, data):
        self.observations[self.step] = data['obs']
        self.actions[self.step] = data['action']
        self.rewards[self.step] = data['reward']
        self.dones[self.step] = data['done']
        self.values[self.step] = data['value']
        self.log_probs[self.step] = data['log_prob']
        self.entropies[self.step] = data['entropy']
        self.step += 1

    def cumpute_returns(self):
        R = 0
        for i, r in enumerate(self.rewards[::-1]):
            if self.dones[-(i+1)]:
                R = r
            else:
                R = r + self.gamma * R
            self.returns[-(i+1)] = R

    def process_rollout(self, last_value, global_step):
        values = np.concatenate([self.values, last_value])
        self.advantages = rlax.truncated_generalized_advantage_estimation(
            self.rewards, self.gamma, self.lam, values
        )
        self.compute_returns()

        # TODO: include logging

        data = (
            self.observations,
            self.actions,
            self.values,
            self.log_probs,
            self.entropies,
            self.advantages,
            self.returns,
        )

        return BufferSamples(data)