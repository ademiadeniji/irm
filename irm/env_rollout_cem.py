from irm.irm import IRM

import torch
import numpy as np

class EnvRolloutCEM(IRM):
    def __init__(self, num_env_skill_rollouts, num_cem_iterations, num_cem_elites, **kwargs):
        super().__init__(**kwargs)
        self.num_env_skill_rollouts = num_env_skill_rollouts
        self.num_cem_iterations = num_cem_iterations
        self.num_cem_elites = num_cem_elites

    def run_skill_selection_method(self):
        best_skill = self.env_rollout_cem().cpu().numpy()
        return [dict(skill=best_skill)]

    def env_rollout_cem(self):
        with torch.no_grad():
            mean = torch.zeros(self.agent.skill_dim, requires_grad=False, device=self.device) + 0.5
            std = torch.zeros(self.agent.skill_dim, requires_grad=False, device=self.device) + 0.25
            for iter in range(self.num_cem_iterations):
                samples = torch.normal(mean.repeat(self.num_env_skill_rollouts, 1), std.repeat(self.num_env_skill_rollouts, 1))
                rewards = []
                for sk in range(self.num_env_skill_rollouts):
                    reward = sum(self.run_skills([samples[sk]])['reward'])
                    if isinstance(reward, float):
                        rewards.append(reward)
                    else:
                        rewards.append(reward)
                sorted_rewards = np.flip(np.argsort(rewards))
                elite_idxs = sorted_rewards[:self.num_cem_elites]
                elites = samples[elite_idxs.copy()]
                mean = torch.mean(elites, dim=0)
                std = torch.std(elites, dim=0)
        return elites[0]