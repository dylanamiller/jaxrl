import time

import jax.numpy as jnp
from jax import grad, jit

from jaxrl.agents.base.pg_base import PGBase
from jaxrl.approx.linear import Linear
from jaxrl.policies import Discrete
from jaxrl.utils import set_seed


class A2C(PGBase):
    def __init__(self, env, approx=Linear, policy=Discrete, seed=0, **kwargs):
        set_seed(seed)

        self.approx = approx(env, kwargs)
        self.policy = policy(env, kwargs)
        self.critic = approx(env, kwargs, out=1)

    def act(self, obs, action=None):
        logits = self.approx(obs)
        action, log_prob, entropy = self.policy(logits)
        value = self.critic(logits)
        return action, value, log_prob, entropy

    def train(self, rollout_data):
        """
        Computes the actor, critic, and entropy losses for training. Uses the stored rollout and the recieved
        rewards, dones, and next_observation from the environment
        """
        # convert lists into torch tensors for ease of manipulation and vectorized operations
        # resultant shape is [seq, batch, ...] meaning [rollout_len, # vector envs, ...]
        # torch stack rollout features
        values = rollout_data['values']
        log_probs_action = rollout_data['log_probs']
        entropies = rollout_data['entropies']
        
        # these are list[list[float|bool]] so shape is [seq, 2]
        # move to whatever device the values are on to support GPU or CPU
        rewards = torch.from_numpy(np.asarray(rewards)).to(values.device)
        dones = torch.from_numpy(np.asarray(dones)).to(values.device)

        # estimate value of next state and TD *without* gradients. This is the target and shouldn't have grads
        with torch.no_grad():
            predictions = network(next_obs)
            next_values = predictions["critic"].squeeze()

            # compute nstep return AKA the temporal difference target
            target_values = self.compute_td_target(next_values, rewards, dones)
            # compute the temporal difference AKA the advantage for each step
            # use .detach() to make sure this does not store gradients, should be covered by torch.no_grad() though
            advantages = target_values - values.detach()
            
            # GOTCHA check, make sure torch does not do any broadcasting and shapes are what we expect
            assert target_values.shape == values.shape
            assert advantages.ndim == 2

        # compute the losses vectorized, shape is [sequence, batch]
        # GOTCHA checks, make sure torch does not do any broadcasting and shapes are what we expect
        # do this before meaning, otherwise the dimensions are lost
        assert log_probs_action.shape == advantages.shape
        policy_loss = -log_probs_action * advantages
        assert policy_loss.ndim == 2
        
        entropy_loss = -entropies * self.entropy_weight
        assert entropy_loss.ndim == 2
        
        # squared error
        assert target_values.shape == values.shape
        value_loss = (target_values - values) ** 2
        assert value_loss.ndim == 2
        
        # mean over seq, batch to create a single loss, critic is half (0.5) of the other losses
        loss = 0.5 * value_loss.mean() + policy_loss.mean() + entropy_loss.mean()
        return loss