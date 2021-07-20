import abc


class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self, n_epochs, rollout_length):
        self.n_epochs = n_epochs
        self.rollout_length = rollout_length

        self._last_obs = None

    @classmethod
    @abc.abstractclassmethod
    def add_experience(self):
        """add necessary transition data to buffer

        Raises:
            NotImplementedError: imlpemented by pg_base/dqn_base
        """
        raise NotImplementedError

    @classmethod
    @abc.abstractclassmethod
    def train(self):
        """update step for agent

        Raises:
            NotImplementedError: implemented by agent
        """
        raise NotImplementedError

