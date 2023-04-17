from irm.irm import IRM

import numpy as np

class RandomSkill(IRM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_skill_selection_method(self):
        return self.random_skill()

    def random_skill(self):
        if len(self.extr_reward) == 1:
            skill = np.random.uniform(0,1,self.agent.skill_dim).astype(np.float32)
            return [dict(skill=skill)]
        else:
            return [dict(skill=np.random.uniform(0,1,self.agent.skill_dim).astype(np.float32)) for _ in range(len(self.extr_reward))]