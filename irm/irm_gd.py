from irm.irm import IRM

import torch

class IRM_GD(IRM):
    def __init__(self, epic_gradient_descent_steps, z_lr, **kwargs):
        self.epic_gradient_descent_steps = epic_gradient_descent_steps
        self.z_lr = z_lr
        super().__init__(**kwargs)

    def run_skill_selection_method(self):
        best_skill = self.irm_gradient_descent().detach().cpu().numpy()
        return [dict(skill=best_skill)]

    def irm_gradient_descent(self, pearson_obs_th=None, pearson_next_obs_th=None, extr_reward_id=0):
        extr_rew = self.get_extr_reward_function(extr_reward_id=extr_reward_id)
        z = torch.full([self.agent.skill_dim], 0.5, requires_grad=True, device=self.device) 
        self.z_optimizer = torch.optim.Adam([z], lr=self.z_lr)
        if self.learnable_reward_scale:
            self.alpha_reward_scale = torch.ones(1, requires_grad=True, device=self.device)
            self.alpha_reward_scale_optimizer = torch.optim.Adam([self.alpha_reward_scale], lr=self.z_lr)
        for step in range(self.epic_gradient_descent_steps):
            metrics = dict()
            if self.matching_metric == "epic":
                loss = self.compute_epic_loss(self.agent.intr_rew, extr_rew, z,
                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th)
            elif self.matching_metric == 'l2':
                loss = self.compute_l2_loss(self.agent.intr_rew, extr_rew, z,
                        pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th)
            elif self.matching_metric == 'l1':
                loss = self.compute_l1_loss(self.agent.intr_rew, extr_rew, z,
                        pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th)
            else:
                raise ValueError("Invalid reward matching metric specified")
            self.z_optimizer.zero_grad()
            if self.learnable_reward_scale:
                self.alpha_reward_scale_optimizer.zero_grad()
            loss.backward()
            self.z_optimizer.step()
            if self.learnable_reward_scale:
                self.alpha_reward_scale_optimizer.zero_grad()
            metrics[f'epic_loss'] = loss
            if self.eval_extr_every_step(self.global_step):
                if extr_reward_id == 0:
                    curr_extr_rew = sum(self.run_skills([z])['reward'])
                metrics['extr_reward'] = curr_extr_rew
            self.logger.log_metrics(metrics, self.global_step, ty='irm')
            if self.print_every_step(self.global_step):
                _, total_time = self.timer.reset()
                with self.logger.log_and_dump_ctx(self.global_step, ty='irm') as log:
                    log('total_time', total_time)
                    log('step', self.global_step)
                    log('extr_reward_id', extr_reward_id)
            self.global_step += 1
        return z