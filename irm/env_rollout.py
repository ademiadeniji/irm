from irm.irm import IRM

class EnvRollout(IRM):
    def __init__(self, num_env_skill_rollouts, **kwargs):
        super().__init__(**kwargs)
        self.num_env_skill_rollouts = num_env_skill_rollouts

    def run_skill_selection_method(self):
        return [dict(skill=self.env_rollout())]

    def env_rollout(self):
        assert len(self.extr_reward) == 1
        
        max_sk, max_rew = None, float('-inf')
        for i in range(self.num_env_skill_rollouts):
            traj_out = self.run_skills([None])
            traj_reward = sum(traj_out['reward'])
            if (max_rew < traj_reward):
                max_rew = traj_reward
                max_sk = traj_out['skill'][0]
        return max_sk