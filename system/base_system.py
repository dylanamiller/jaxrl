import abc

class BaseSystem(metaclass=abc.ABCMeta):
    def __init__(self, env, agent, rollout_length, total_steps=1e6):
        self.env = env
        self.agent = agent
        self.rollout_length = rollout_length
        self.total_steps = total_steps
        
        self._last_obs = self.env.reset()

    @classmethod
    @abc.abstractclassmethod
    def run(self):
        """run system for chosen number of steps

        Raises:
            NotImplementedError: imlpemented by system/ma_system
        """
        raise NotImplementedError

                