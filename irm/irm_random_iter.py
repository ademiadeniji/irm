from irm.irm import IRM
from irm.irm_rand import IRM_Rand

import torch
import numpy as np

class IRM_RandomIter(IRM_Rand):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run_skill_selection_method(self):
        best_skills = self.irm_random_iter()
        return [dict(skill=sk) for sk in best_skills]

    def irm_random_iter(self):   
        best_skills = []
        tl_lst = [self.agent.skill_duration for _ in range(self.n_rewards)]

        # select first best skill with cfg's pearson / canonical settings
        min_sk = self.irm_rand_search(extr_reward_id=0).cpu().numpy()
        best_skills.append(min_sk)

        # rollout first skill. select second best skill with possibly new canonical setting
        for r in range(1, self.n_rewards):
            prev_rollout = self.run_skills(best_skills, tl_lst[:r], use_handcrafted=True)
            pearson_obs_th, pearson_next_obs_th = self.get_pearson_samples_iter(prev_rollout, r-1) # obs from prev rollout
            
            # take care of prev_rollout_last
            min_sk = self.irm_rand_search(pearson_obs_th, pearson_next_obs_th, r).cpu().numpy()
            best_skills.append(min_sk)

        return best_skills

    def get_pearson_samples_iter(self, out, r):
        if self.pearson_setting_sequencing == "prev_rollout":
            epic_obs = np.array(out['ep_obs'][max(r-1, 0)])
            epic_actions = np.array(out['ep_action'][max(r-1, 0)])
        elif self.pearson_setting_sequencing == "prev_rollout_last":
            epic_obs = np.array(out['ep_obs'][max(r-1, 0)])
            epic_actions = np.array(out['ep_action'][max(r-1, 0)])
            if r > 1:
                epic_obs = np.tile(epic_obs[epic_obs.shape[0] // 2:], (2, 1))
                epic_actions = np.tile(epic_actions[epic_actions.shape[0] // 2:], (2, 1))
        else:
            epic_obs = np.array(out['ep_obs'][r])
            epic_actions = np.array(out['ep_action'][r])
        obs, next_obs = torch.from_numpy(epic_obs[:-1, :]).to(self.device), torch.from_numpy(epic_obs[1:, :]).to(self.device)
        return obs, next_obs