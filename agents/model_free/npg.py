import time

from scipy.stats import entropy

import jax.numpy as jnp
from jax import grad, jit

from jaxrl.agents.base.pg_base import PGBase
from jaxrl.approx.linear import Linear
from jaxrl.policies import Discrete
from jaxrl.utils import set_seed, mean_kl_div, likelihood_ratio


class NPG(PGBase):
    def __init__(self, env, approx=Linear, policy=Discrete, cg_iters=10, damping=1e-4, seed=0, **kwargs):
        set_seed(seed)

        self.approx = approx(env, kwargs)
        self.old_approx = approx(env, kwargs)
        self.policy = policy(env, kwargs)
        self.critic = approx(env, kwargs, out=1)

        self.cg_iters = cg_iters
        self.damping = damping

    def act(self, obs, action=None):
        logits = self.approx(obs)
        action, log_prob = self.policy(logits)
        value = self.critic(obs)

        return action, value, log_prob, entropy(logits)

    def cpi_surrogate(self, observations, actions, advantages):
        """conservative policy iteration to limit policy update

        Args:
            observations ([type]): [description]
            actions ([type]): [description]
            advantages ([type]): [description]

        Returns:
            [type]: [description]
        """
        old_dist = self.old_approx.dist(observations, actions)
        new_dist = self.approx.dist(observations, actions)
        LR = likelihood_ratio(new_dist, old_dist)
        surr = jnp.mean(LR*advantages)
        return surr

    def flat_vpg(self, observations, actions, advantages):
        cpi_surr = grad(self.cpi_surrogate)(observations, actions, advantages)
        vpg_grad = grad(cpi_surr, self.approx.trainable_params)
        vpg_grad = jnp.concatenate([g.contiguous().view(-1).data.numpy() for g in vpg_grad])
        return vpg_grad

    @jit
    def cg_solve(f_Ax, b, x_0=None, cg_iters=10, residual_tol=1e-10):
        """conjugate gradients to solve for fvp

        Args:
            f_Ax ([type]): [description]
            b ([type]): [description]
            x_0 ([type], optional): [description]. Defaults to None.
            cg_iters (int, optional): [description]. Defaults to 10.
            residual_tol ([type], optional): [description]. Defaults to 1e-10.

        Returns:
            [type]: [description]
        """

        x = jnp.zeros_like(b) #if x_0 is None else x_0
        r = b.copy() #if x_0 is None else b-f_Ax(x_0)
        p = r.copy()
        rdotr = r.dot(r)

        for i in range(cg_iters):
            z = f_Ax(p)
            v = rdotr / p.dot(z)
            x += v * p
            r -= v * z
            newrdotr = r.dot(r)
            mu = newrdotr / rdotr
            p = r + mu * p

            rdotr = newrdotr
            if rdotr < residual_tol:
                break

        return x
        
    def fvp(self, observations, actions, vector, regu_coef=None):
        """fisher vector product

        Args:
            observations ([type]): [description]
            actions ([type]): [description]
            vector ([type]): [description]
            regu_coef ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        
        regu_coef = self.damping if regu_coef is None else regu_coef
        if self.hvp_subsample is not None and self.hvp_subsample < 0.99:
            num_samples = observations.shape[0]
            rand_idx = jnp.random.choice(num_samples, size=int(self.hvp_subsample*num_samples))
            obs = observations[rand_idx]
            act = actions[rand_idx]
        else:
            obs = observations
            act = actions
        old_dist = self.old_approx.dist(obs, act)
        new_dist = self.approx.dist(obs, act)
        mean_kl = mean_kl_div(new_dist, old_dist)
        grad_fo = grad(mean_kl, self.policy.trainable_params, create_graph=True)
        flat_grad = jnp.concatenate([g.contiguous().view(-1) for g in grad_fo])
        h = jnp.sum(flat_grad*vector)
        hvp = grad(h, self.approx.trainable_params)
        hvp_flat = jnp.concatenate([g.contiguous().view(-1).data.numpy() for g in hvp])
        return hvp_flat + regu_coef*vector

    def build_fvp_eval(self, inputs, regu_coef=None):
        def eval(v):
            full_inp = inputs + [v] + [regu_coef]
            fvp = self.fvp(*full_inp)
            return fvp
        return eval

    # ----------------------------------------------------------
    def train(self, rollout_data):
        rollout_data = self.process_rollouts()
        observations = rollout_data.observations
        actions = rollout_data.actions
        advantages = rollout_data.advantages
        rewards = rollout_data.rewards

        # Keep track of times for various computations
        t_gLL = 0.0
        t_FIM = 0.0

        # Optimization algorithm
        # --------------------------
        surr_before = self.cpi_surrogate(observations, actions, advantages).data.numpy().ravel()[0]

        # VPG
        ts = time.time()
        vpg_grad = self.flat_vpg(observations, actions, advantages)
        t_gLL += time.time() - ts

        # NPG
        ts = time.time()
        hvp = self.build_fvp_eval([observations, actions],
                                  regu_coef=self.damping)
        npg_grad = self.cg_solve(hvp, vpg_grad, x_0=vpg_grad.copy(),
                            cg_iters=self.cg_iters)
        t_FIM += time.time() - ts

        # Step size computation
        # --------------------------
        if self.alpha is not None:
            alpha = self.alpha
            n_step_size = (alpha ** 2) * jnp.dot(vpg_grad.T, npg_grad)
        else:
            n_step_size = self.n_step_size
            alpha = jnp.sqrt(jnp.abs(self.n_step_size / (jnp.dot(vpg_grad.T, npg_grad) + 1e-20)))

        # Policy update
        # --------------------------
        curr_params = self.approx.get_param_values()
        new_params = curr_params + alpha * npg_grad
        self.approx.set_param_values(new_params)
        surr_after = self.cpi_surrogate(observations, actions, advantages).data.numpy().ravel()[0]
        old_dist = self.old_approx.dist(observations, actions)
        new_dist = self.approx.dist(observations, actions)
        kl_dist = mean_kl_div(new_dist, old_dist)
        self.old_approx.set_param_values(new_params)

        # Log information
        if self.save_logs:
            self.logger.log_kv('alpha', alpha)
            self.logger.log_kv('delta', n_step_size)
            self.logger.log_kv('time_vpg', t_gLL)
            self.logger.log_kv('time_npg', t_FIM)
            self.logger.log_kv('kl_dist', kl_dist)
            self.logger.log_kv('surr_improvement', surr_after - surr_before)
            self.logger.log_kv('running_score', self.running_score)

        return base_stats

