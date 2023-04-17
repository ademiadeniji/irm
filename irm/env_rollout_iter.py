from irm.irm import IRM

class EnvRolloutIter(IRM):
    def __init__(self, num_env_skill_rollouts, **kwargs):
        super().__init__(**kwargs)
        self.num_env_skill_rollouts = num_env_skill_rollouts

    def run_skill_selection_method(self):
        best_skills = self.env_rollout_iter()
        return [dict(skill=sk) for sk in best_skills]

    def env_rollout_iter(self):   
        best_skills = [] 
        tl_lst = [self.agent.skill_duration for _ in range(self.n_rewards)]
        extr_reward_lst = range(self.n_rewards)
        num_rollouts_per_skill = self.num_env_skill_rollouts // self.n_rewards

        for r in range(self.n_rewards):
            max_sk, max_rew = None, float('-inf')
            for i in range(num_rollouts_per_skill):
                traj_out = self.run_skills(best_skills + [None], tl_lst[:r+1], use_handcrafted=True)
                if (max_rew < traj_out['reward'][r]):
                    max_rew = traj_out['reward'][r] 
                    max_sk = traj_out['skill'][r] 
            best_skills.append(max_sk)

        return best_skills