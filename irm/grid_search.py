from irm.irm import IRM

import numpy as np
import math

class GridSearch(IRM):
    def __init__(self, grid_search_size, **kwargs):
        super().__init__(**kwargs)
        self.grid_search_size = grid_search_size

    def run_skill_selection_method(self):
        assert len(self.extr_reward) == 1
        return [dict(skill=self.grid_search())]
        
    def grid_search(self):
        max_sk, max_rew = None, float('-inf')
        curr_sk = np.zeros(self.agent.skill_dim, dtype=np.float32)
        for i in range(math.floor(1 / self.grid_search_size) + 1):
            traj_out = self.run_skills([curr_sk])
            traj_reward = sum(traj_out['reward'])
            if (max_rew < traj_reward):
                max_rew = traj_reward 
                max_sk = traj_out['skill'][0]
        return max_sk