from jaxrl.system.base_system import BaseSystem
from jaxrl.buffers.buffer import Buffer

from torch.utils.tensorboard import SummaryWriter

import jax.numpy as jnp

class MASystem(BaseSystem):
    def __init__(self, env, agent, rollout_length, total_steps=1e6):
        super().__init__(env, agent, rollout_length, total_steps)

        self.agent_ids = env.possible_agents
        self.n_agents = len(self.agent_ids)

        self.agents = {agent: agent(env, agent) for agent in self.agent_ids}
        self.buffers = {agent: Buffer(env, agent) for agent in self.agent_ids}

        self.writer = SummaryWriter(f"runs/{experiment_name}")
        self.writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
            '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))
            )

    @classmethod
    def run(self, print_freq=10):
        global_step = 0
        n_updates = self.total_steps // self.rollout_length

        for update in range(1, n_updates+1):
            G = {agent: list() for agent in self.agent_ids}
            g = {agent: 0 for agent in self.agent_ids}
            episode_length = list()
            epl = 0

            for step in range(self.rollout_length):
                actions = dict()
                for agent in self.agent_ids:
                    action, value, log_prob, entropy = self.agents[agent].act(self._last_obs[agent])
                    actions[agent] = action

                    self.buffers[agent].add.exp(
                        step,
                        {
                            'action': action,
                            'value': value,
                            'log_prob': log_prob,
                            'entropy': entropy,
                        }
                    )

                next_obs, rewards, dones, infos = self.env.step(actions)

                for agent in self.agent_ids:
                    self.buffers[agent].add_exp(
                        step,
                        {
                            'obs': self._last_obs[agent],
                            'reward': rewards[agent],
                            'done': dones[agent],
                        }
                    )

                    g[agent] += rewards[agent]
                    d_ = False

                    if dones[agent]:
                        d_ = True

                        G[agent].append(g[agent]) # dense reward
                        # G[agent].append(rs[agent])  # sparse reward
                        g[agent] = 0

                    if d_:
                        next_obs = self.env.reset()
                        episode_length.append(epl)
                        epl = 0

                self._last_obs = next_obs
                global_step += 1

            epl += 1

            for agent in self.agent_ids:
                self.buffers[agent].process_rollout(global_step)
                self.agents[agent].train(self.buffers[agent].rollout_data)
                self.buffers[agent].reset()

                if update % print_freq == 0:
                    print(f"{agent}/epoch_rewards_mean:", jnp.mean(G[agent]))

                self.writer.add_scalar(f"returns/{agent}/epoch_rewards", jnp.sum(G[agent]), global_step)
                self.writer.add_scalar(f"returns/{agent}/epoch_rewards_max", jnp.max(G[agent]), global_step)
                self.writer.add_scalar(f"returns/{agent}/epoch_rewards_min", jnp.min(G[agent]), global_step)
                self.writer.add_scalar(f"returns/{agent}/epoch_rewards_mean", jnp.mean(G[agent]), global_step)
                self.writer.add_scalar(f"returns/{agent}/epoch_rewards_std", jnp.std(G[agent]), global_step)