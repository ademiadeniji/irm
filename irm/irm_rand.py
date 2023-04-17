from irm.irm import IRM

import torch

class IRM_Rand(IRM):
    def __init__(self, num_random_skills, **kwargs):
        super().__init__(**kwargs)
        self.num_random_skills = num_random_skills

    def run_skill_selection_method(self):
        assert len(self.extr_reward) == 1
        best_skill = self.irm_rand_search().cpu().numpy()
        return [dict(skill=best_skill)]

    def irm_rand_search(self, pearson_obs_th=None, pearson_next_obs_th=None, extr_reward_id=0):
        """
        Optional arguments used for IRM Random Iter (skill selection for sequential goals)
        """
        extr_rew = self.get_extr_reward_function(extr_reward_id=extr_reward_id) 
        min_sk, min_loss = None, float('inf')
        for i in range(self.num_random_skills):
            metrics = dict()
            if self.learnable_reward_scale:
                self.alpha_reward_scale = torch.rand((1), device=self.device) * 100
            skill = torch.rand((self.agent.skill_dim), device=self.device)
            with torch.no_grad():
                if self.matching_metric == "epic":
                    loss = self.compute_epic_loss(self.agent.intr_rew, extr_rew, skill,
                                        pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th)
                elif self.matching_metric == 'l2':
                    loss = self.compute_l2_loss(self.agent.intr_rew, extr_rew, skill, 
                                        pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th)
                elif self.matching_metric == 'l1':
                    loss = self.compute_l1_loss(self.agent.intr_rew, extr_rew, skill,
                                        pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th)
                else:
                    raise ValueError("Invalid reward matching metric specified")
                if (loss < min_loss):
                    min_loss = loss 
                    min_sk = skill 
            metrics['epic_loss'] = min_loss 
            if self.eval_extr_every_step(self.global_step):
                if extr_reward_id == 0:
                    curr_extr_rew = sum(self.run_skills([min_sk])['reward'])
                metrics['extr_reward'] = curr_extr_rew
            self.logger.log_metrics(metrics, self.global_step, ty='irm')
            if self.print_every_step(self.global_step):
                _, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='irm') as log:
                    log('total_time', total_time)
                    log('step', self.global_step)
                    log('extr_reward_id', extr_reward_id)
            self.global_step += 1
        return min_sk