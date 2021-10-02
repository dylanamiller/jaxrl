import haiku as hk


class MuZeroModel:
    def __init__(self, config):
        self.representation_net = hk.nets.MLP(config['representation_net']['layers'])
        self.prediction_net = hk.nets.MLP(config['prediction_net']['layers'])
        self.dynamics_net = hk.nets.MLP(config['dynamics_net']['layers'])

    def initial_inference(self):
        """ h: o -> s, f: s -> p, v
        """

        pass

    def recurrent_inference(self):
        """ g: s -> s, f: s -> p, v
        """

        pass

