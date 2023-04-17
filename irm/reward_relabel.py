from irm.irm import IRM

class RewardRelabel(IRM):
    def __init__(self, update_skill_every_step, **kwargs):
        super().__init__(**kwargs)
        self.update_skill_every_step = update_skill_every_step

    def run_skill_selection_method(self):
        return self.reward_relabel()

    def reward_relabel(self):
        print("Loading pretraining data...")
        self.replay_buffer._load()
        print("Finished loading pretraining data.")
        with torch.no_grad():
            max_sk, max_rew = None, float('-inf')
            for path, ep in self.replay_buffer._episodes.items():
                obs_copy = ep['observation'].copy()
                skill_copy = ep['skill'].copy()
                if (ep['observation'].shape[0]-1) % self.update_skill_every_step != 0:
                    obs_copy = np.concatenate((obs_copy, np.expand_dims(obs_copy[-1], 0)), 0)
                    skill_copy = np.concatenate((skill_copy, np.expand_dims(skill_copy[-1], 0)), 0)
                obs, _ = self.agent.process_observation(obs_copy[:-1])
                next_obs, _ = self.agent.process_observation(obs_copy[1:])
                skill = skill_copy[1:]
                extr_rew_fn = self.get_extr_reward_function()
                reward = extr_rew_fn(torch.from_numpy(obs).to(self.device), torch.from_numpy(next_obs).to(self.device), skill)
                reward = reward.squeeze(-1).reshape(-1, self.update_skill_every_step)
                reward = torch.sum(reward, -1)
                rew, sk_idx = torch.max(reward, 0)
                if rew > max_rew:
                    max_sk = skill[self.update_skill_every_step*sk_idx.item()]
                    max_rew = rew
            return [dict(skill=max_sk)]
            # print(f"Best skill in pretraining replay buffer: {max_sk}")