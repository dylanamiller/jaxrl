from jaxrl.system.base_system import BaseSystem


class MASystem(BaseSystem):
    def __init__(self, env, agents, rollout_length, total_steps=1e6):
        super().__init__(env, agents, rollout_length, total_steps)

    def run(self):
        
