import abc

class BaseSystem(metaclass=abc.ABCMeta):
    def __init__(self, env, agent, rollout_length, total_steps=1e6):
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length
        self.total_steps = total_steps
        
        self._last_obs = self.env.reset()

    def run(self):

        n_updates = self.total_steps // self.rollout_length
        for update in range(n_updates):

            for step in range(self.rollout_length):

                