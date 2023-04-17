from irm.irm import IRM

import torch
import numpy as np

class IRM_CEM(IRM):
    def __init__(self, num_cem_iterations, num_cem_samples, num_cem_elites, **kwargs):
        self.num_cem_iterations = num_cem_iterations
        self.num_cem_samples = num_cem_samples
        self.num_cem_elites = num_cem_elites
        super().__init__(**kwargs)

    def run_skill_selection_method(self):
        best_skill = self.irm_cem().cpu().numpy()
        return [dict(skill=best_skill)]

    def irm_cem(self, pearson_obs_th=None, pearson_next_obs_th=None, extr_reward_id=0):
        extr_rew = self.get_extr_reward_function(extr_reward_id=extr_reward_id)
        if self.learnable_reward_scale:
            with torch.no_grad():
                mean = torch.zeros(self.agent.skill_dim+1, requires_grad=False, device=self.agent.device) + 0.5
                std = torch.zeros(self.agent.skill_dim+1, requires_grad=False, device=self.agent.device) + 0.25
                for idx in range(self.num_cem_iterations):
                    samples = torch.normal(mean.repeat(self.num_cem_samples, 1), std.repeat(self.num_cem_samples, 1))
                    if self.learnable_reward_scale:
                        skill_samples = samples[:, :-1]
                        scale_samples = samples[:, -1]
                    losses = []
                    for sk in range(self.num_cem_samples):
                        if self.learnable_reward_scale:
                            self.alpha_reward_scale = scale_samples[sk]
                        if self.matching_metric == "epic":
                            losses.append(self.compute_epic_loss(self.agent.intr_rew, extr_rew, skill_samples[sk],
                                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th).item())
                        elif self.matching_metric == 'l2':
                            losses.append(self.compute_l2_loss(self.agent.intr_rew, extr_rew, skill_samples[sk],
                                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th).item())
                        elif self.matching_metric == 'l1':
                            losses.append(self.compute_l1_loss(self.agent.intr_rew, extr_rew, skill_samples[sk],
                                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th).item())
                        else:
                            raise ValueError("Invalid reward matching metric specified")
                    sorted_losses = np.argsort(losses)
                    elite_idxs = sorted_losses[:self.num_cem_elites]
                    elites = samples[elite_idxs]
                    mean = torch.mean(elites, dim=0)
                    std = torch.std(elites, dim=0)
                    metrics = self.calculate_metrics(extr_reward_id, mean, std, losses, sorted_losses)
                    self.logger.log_metrics(metrics, self.global_step, ty='irm')
                    if self.print_every_step(self.global_step):
                        _, total_time = self.timer.reset()
                        with self.logger.log_and_dump_ctx(self.global_step, ty='irm') as log:
                            log('total_time', total_time)
                            log(f'step', self.global_step)
                            log(f'extr_reward_id', extr_reward_id)
            return elites[0][:-1]
        else:
            with torch.no_grad():
                mean = torch.zeros(self.agent.skill_dim, requires_grad=False, device=self.device) + 0.5
                std = torch.zeros(self.agent.skill_dim, requires_grad=False, device=self.device) + 0.25
                for idx in range(self.num_cem_iterations):
                    samples = torch.normal(mean.repeat(self.num_cem_samples, 1), std.repeat(self.num_cem_samples, 1))
                    losses = []
                    for sk in range(self.num_cem_samples):
                        if self.matching_metric == "epic":
                            losses.append(self.compute_epic_loss(self.agent.intr_rew, extr_rew, samples[sk],
                                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th).item())
                        elif self.matching_metric == 'l2':
                            losses.append(self.compute_l2_loss(self.agent.intr_rew, extr_rew, samples[sk],
                                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th).item())
                        elif self.matching_metric == 'l1':
                            losses.append(self.compute_l1_loss(self.agent.intr_rew, extr_rew, samples[sk],
                                            pearson_obs_th=pearson_obs_th, pearson_next_obs_th=pearson_next_obs_th).item())
                        else:
                            raise ValueError("Invalid reward matching metric specified")
                    sorted_losses = np.argsort(losses)
                    elite_idxs = sorted_losses[:self.num_cem_elites]
                    elites = samples[elite_idxs]
                    mean = torch.mean(elites, dim=0)
                    std = torch.std(elites, dim=0)
                    metrics = self.calculate_metrics(extr_reward_id, mean, std, losses, sorted_losses)
                    if self.eval_extr_every_step(self.global_step):
                        if extr_reward_id == 0:
                            curr_extr_rew = sum(self.run_skills([elites[0]])['reward'])
                        metrics['extr_reward'] = curr_extr_rew
                        print(curr_extr_rew)
                    self.logger.log_metrics(metrics, self.global_step, ty='irm')
                    if self.print_every_step(self.global_step):
                        _, total_time = self.timer.reset()
                        with self.logger.log_and_dump_ctx(self.global_step, ty='irm') as log:
                            log('total_time', total_time)
                            log(f'step', idx)
                            log(f'extr_reward_id', extr_reward_id)
                    self.global_step += 1
            return elites[0]

    def calculate_metrics(self, extr_reward_id, mean, std, losses, sorted_losses):
        losses = np.array(losses)
        metrics = dict()
        metrics[f'cem_mean_0'] = mean[0]
        metrics[f'cem_std_0'] = std[0]
        metrics[f'cem_mean_0'] = mean[1]
        metrics[f'cem_std_0'] = std[1]
        metrics[f'mean_loss'] = np.mean(losses)
        metrics[f'mean_elite_loss'] = np.mean(losses[sorted_losses[:self.num_cem_elites]])
        metrics[f'epic_loss'] = losses[sorted_losses[0]]
        return metrics