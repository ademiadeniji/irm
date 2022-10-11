import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import fetch
import utils
from logger import Logger
from video import TrainVideoRecorder, VideoRecorder
import wandb

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
import envs.plane as plane
import matplotlib.pyplot as plt



def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        torch.cuda.set_device(cfg.device)
        
        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([cfg.experiment,cfg.agent.name,cfg.domain,cfg.obs_type,str(cfg.seed)])
            wandb.init(project="urlb",group=cfg.agent.name,name=exp_name)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb,use_wandb=cfg.use_wandb)
        # create envs
        time_limit = cfg.time_limit if cfg.use_time_limit else None
        if 'plane' in cfg.domain:
            self.train_env = plane.make(cfg.domain, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed, time_limit)
            self.bounds = dict(min=-128, max=128)
        elif 'fetch' in cfg.domain:
            self.train_env = fetch.make(cfg.domain, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.random_reset, time_limit=time_limit)
            self.eval_env = fetch.make(cfg.domain, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.random_reset, time_limit=time_limit)
            self.bounds = dict(min=0.5, max=1.6)
        else: 
            # DMC setup
            task = PRIMAL_TASKS[self.cfg.domain]
            self.train_env = dmc.make(task, cfg.obs_type,
                                    cfg.frame_stack, cfg.action_repeat,
                                    cfg.seed, cfg.use_custom, time_limit=time_limit,
                                    random_reset=cfg.random_reset)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent)

        # initialize from pretrained
        if cfg.restore_snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()['agent']
            self.agent.init_from(pretrained_agent)
            print(f"initialized agent from {self.cfg.restore_snapshot_dir}")
        else:
            print(f"randomly initialized agent")

        # create video recorders
        if "fetch" in self.cfg.domain:
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb, frame_lst=True)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb, frame_lst=True)
        else: 
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None,
                camera_id= 0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)


        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.device = self.cfg.device
        self.discount = self.cfg.discount

        self.plot_2d, self.draw_plot = True, True
        if self.cfg.domain == "plane":
            self.xlim1, self.ylim1 = -140, -140
            self.xlim2, self.ylim2 = 140, 140
            self.bounds = dict(min=-128, max=128)
            self.train_env.step_limit = time_limit if cfg.use_time_limit else 200
        elif self.cfg.domain == "fetch_push":
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.train_env.step_limit = self.train_env.env._max_episode_steps
        elif self.cfg.domain == "fetch_barrier":
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.train_env.step_limit = self.train_env.env._max_episode_steps
            min_th, max_th = torch.zeros(1, 28).to(cfg.device), torch.ones(1, 28).to(cfg.device)
            min_th[:, 3] = 0.96
            min_th[:, 4] = 0.5
            max_th[:, 3] = 1.4
            max_th[:, 4] = 1
            self.bounds = {'min': min_th, 'max': max_th}
        elif self.cfg.domain == "fetch_reach":
            self.xlim1, self.ylim1 = 0.7, 0.5
            self.xlim2, self.ylim2 = 1.7, 1.2
            self.plot_2d = False 
            self.train_env.step_limit = 200
        elif self.cfg.domain == "jaco":
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 1, 1
            self.bounds = dict(min=-1, max=1)
            if self.cfg.agent.extr_reward == "goal_np75_p5_p5":
                self.plot_2d = False
            self.train_env.step_limit = 125
        elif self.cfg.domain == "walker":
            self.xlim1, self.ylim1 = -1, -1
            self.xlim2, self.ylim2 = 0, 1
            self.bounds = dict(min=-1, max=1)
            self.train_env.step_limit = self.cfg.time_limit * 4
            self.train_env.step_limit = 1000 # hardcoded
        else:
            self.draw_plot = False

        if self.draw_plot:
            self.plot_dir = self.work_dir / 'plot'
            self.plot_dir.mkdir(exist_ok=True)

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot_dir = Path(self.cfg.restore_snapshot_dir)
        snapshot = snapshot_dir / f'snapshot_{self.cfg.restore_snapshot_ts}.pt'

        with snapshot.open('rb') as f:
            payload = torch.load(f)
        return payload

    def epic_heat_plot(self, PEARSON_SETTING, CANONICAL_SETTING):
        # environment settings
        NUM_DISCRETIZE = 5 
        
        # skills to traverse
        axis_labels = [i/NUM_DISCRETIZE for i in range(NUM_DISCRETIZE+1)]
        skills_to_traverse = [np.array([i, j]).astype(np.float32) for i in axis_labels for j in axis_labels]
        
        # infra setup
        time_step = self.train_env.reset() 
        episode_step = 0
        skills = []
        rewards = []
        heatmap = np.zeros((NUM_DISCRETIZE+1, NUM_DISCRETIZE+1))

        for m in range(len(skills_to_traverse)):
            sk = skills_to_traverse[m]
            out = self.run_skill(sk)
            self._global_step += out['step']
            self._global_episode += 1
            skills.append(out['skill'])

            with torch.no_grad():
                epic_loss = self.compute_epic_loss_ft(self.agent.compute_inner_product, self.agent.get_extr_rew(), np.array(out['ep_obs']), out['skill'], np.array(out['ep_action']), PEARSON_SETTING, CANONICAL_SETTING)
            rewards.append(-epic_loss.cpu().numpy()) # note the negative sign
            heatmap[round(sk[0]*NUM_DISCRETIZE), round(sk[1]*NUM_DISCRETIZE)] += epic_loss.cpu().numpy()


        import seaborn as sns
        from matplotlib.patches import Rectangle
        fig, ax1 = plt.subplots(1, 1)
        ax = sns.heatmap(heatmap, linewidth=0.5, xticklabels=axis_labels, yticklabels=axis_labels).set(title=f"epic losses for go right")
        plt.show()
        plot_name = f"{self.plot_dir}/epic_heat{self.cfg.seed}.png"
        plt.savefig(plot_name)
        plt.clf() 
        print(f"saved in {plot_name}")

    def rew_epic_plot(self, PEARSON_SETTING, CANONICAL_SETTING, same_batch=False):
        # skills to traverse
        NUM_DISCRETIZE = 100
        skills_to_traverse = [None for i in range(NUM_DISCRETIZE)]
        
        # infra setup
        time_step = self.train_env.reset() 
        episode_step = 0
        skills = []
        epic_losses = []
        extr_rewards = []
        all_ep_obs = []

        # run skills
        for m in range(len(skills_to_traverse)):
            sk = skills_to_traverse[m]
            out = self.run_skill(sk)
            self._global_step += out['step']
            self._global_episode += 1
            skills.append(out['skill'])
            with torch.no_grad():
                epic_loss = self.compute_epic_loss(np.array(out['ep_obs']), out['skill'], np.array(out['ep_action']), PEARSON_SETTING, CANONICAL_SETTING, same_batch)
            epic_losses.append(epic_loss.item())
            extr_rewards.append(out['reward'].cpu()) 
            all_ep_obs.append(out['ep_obs'])
            
        # scatter plot
        BATCH_STR = "1batch" if same_batch else "multibatch"
        title = f"{PEARSON_SETTING} Pearson Samples and {CANONICAL_SETTING} Canonical Samples w/ {BATCH_STR}"
        plot_name = f"P_{PEARSON_SETTING}_C_{CANONICAL_SETTING}_{BATCH_STR}"
        self.scatter_plot(epic_losses, extr_rewards, title, plot_name, xlabel="Epic Loss", ylabel="Extrinsic Reward")

        # plot lowest epic loss trajectories
        if self.draw_plot:
            SKILLS_TO_GRAPH = 3
            sort_epic = np.argsort(np.array(epic_losses))
            low_epic = sort_epic[:SKILLS_TO_GRAPH]
            low_style_map = ['green', 'lawngreen', 'aquamarine', 'deepskyblue',  'blue', 'darkblue',
                        'slateblue', 'rebeccapurple']
            high_epic = sort_epic[-SKILLS_TO_GRAPH:]
            high_style_map = ['maroon', 'orangered', 'darkorange', 'gold', 'yellow', 'mediumorchid',
                            'violet', 'magenta']
            style_map = [*low_style_map[:SKILLS_TO_GRAPH], *high_style_map[:SKILLS_TO_GRAPH]]

            skills_to_plot = [*[skills[i] for i in low_epic], *[skills[i] for i in high_epic]]
            traj_to_plot = [*[all_ep_obs[i] for i in low_epic], *[all_ep_obs[i] for i in high_epic]]
            labels_to_plot = [*[f"epic loss={epic_losses[i]:.2f}" for i in low_epic], *[f"epic loss={epic_losses[i]:.2f}" for i in high_epic]]
            title = f"Best Traj via EPIC loss w/ {PEARSON_SETTING} Pearson and {CANONICAL_SETTING} Canonical w/ {BATCH_STR}"
            name = f"P_{PEARSON_SETTING}_C_{CANONICAL_SETTING}_{BATCH_STR}_traj"
            self.plot_traj(skills_to_plot, traj_to_plot, name, style_map=style_map, title=title, goal=self.agent.get_goal(), labels=labels_to_plot)

    def sequential_rew_epic_plot(self, PEARSON_SETTING, CANONICAL_SETTING, same_batch=False):
        # skills to traverse
        NUM_DISCRETIZE = 100
        skills_to_traverse = [None for i in range(NUM_DISCRETIZE)]
        n_rewards = len(self.cfg.agent.extr_reward_seq)
        env_tl = self.train_env.step_limit
        
        # infra setup
        time_step = self.train_env.reset() 
        episode_step = 0
        skills = [[] for _ in range(n_rewards)]
        epic_losses = [[] for _ in range(n_rewards)]
        extr_rewards = [[] for _ in range(n_rewards)]
        all_ep_obs = [[] for _ in range(n_rewards)]
        best_skills = []
        best_skill_ids = []

        # run skills
        for r in range(n_rewards):
            self.agent.extr_reward_id = r
            for m in range(len(skills_to_traverse)):
                # run any prior skills 
                sk = skills_to_traverse[m]
                skill_lst = best_skills + [sk]
                tl_lst = [env_tl // n_rewards for _ in range(r+1)]
                out = self.run_skills(skill_lst, tl_lst)
                if len(out['ep_obs'][r]) <= 1: # hack for fetch primitives resets
                    continue
                with torch.no_grad():
                    if PEARSON_SETTING == "prev_rollout":
                        epic_obs = np.array(out['ep_obs'][max(r-1, 0)])
                        epic_actions = np.array(out['ep_action'][max(r-1, 0)])
                    if PEARSON_SETTING == "prev_rollout_last":
                        epic_obs = np.array(out['ep_obs'][max(r-1, 0)])
                        epic_actions = np.array(out['ep_action'][max(r-1, 0)])
                        if r > 1:
                            epic_obs = np.tile(epic_obs[epic_obs.shape[0] // 2:], (2, 1))
                            epic_actions = np.tile(epic_actions[epic_actions.shape[0] // 2:], (2, 1))
                    else:
                        epic_obs = np.array(out['ep_obs'][r])
                        epic_actions = np.array(out['ep_action'][r])
                    epic_loss = self.compute_epic_loss(epic_obs, out['skill'][r], epic_actions, PEARSON_SETTING, CANONICAL_SETTING, same_batch)
                epic_losses[r].append(epic_loss.item())
                extr_rewards[r].append(out['reward'][r].cpu()) 
                skills[r].append(out['skill'][r])
                all_ep_obs[r].append(out['ep_obs'])
            
            # scatter plot 
            BATCH_STR = "1batch" if same_batch else "multibatch"
            title = f"{PEARSON_SETTING} Pearson Samples and {CANONICAL_SETTING} Canonical Samples w/ {BATCH_STR}"
            plot_name = f"rew{r}_P_{PEARSON_SETTING}_C_{CANONICAL_SETTING}_{BATCH_STR}"
            self.scatter_plot(epic_losses[r], extr_rewards[r], title, plot_name, xlabel="Epic Loss", ylabel="Extrinsic Reward")

            # plot traj
            if self.draw_plot:
                SKILLS_TO_GRAPH, N_SKIP = 3, 3
                sort_epic = np.argsort(np.array(epic_losses[r]))
                low_epic = sort_epic[:SKILLS_TO_GRAPH*N_SKIP:N_SKIP] # slightly more randomness here
                low_style_map = ['green', 'lawngreen', 'aquamarine', 'deepskyblue',  'blue', 'darkblue',
                            'slateblue', 'rebeccapurple']
                high_epic = sort_epic[-SKILLS_TO_GRAPH*N_SKIP::N_SKIP]
                high_style_map = ['maroon', 'orangered', 'darkorange', 'gold', 'yellow', 'mediumorchid',
                                'violet', 'magenta']
                style_map = [*low_style_map[:SKILLS_TO_GRAPH], *high_style_map[:SKILLS_TO_GRAPH]]

                graph_skills = np.array(skills[r])
                graph_ep_obs = np.array(all_ep_obs[r])
                graph_ep_obs = graph_ep_obs.reshape(graph_ep_obs.shape[0], -1, *graph_ep_obs.shape[3:])
                skills_to_plot = [*[graph_skills[i] for i in low_epic], *[graph_skills[i] for i in high_epic]]
                traj_to_plot = [*[graph_ep_obs[i] for i in low_epic], *[graph_ep_obs[i] for i in high_epic]]
                labels_to_plot = [*[f"epic loss={epic_losses[r][i]:.2f}" for i in low_epic], *[f"epic loss={epic_losses[r][i]:.2f}" for i in high_epic]]
                title = f"Best Traj via EPIC loss w/ {PEARSON_SETTING} Pearson and {CANONICAL_SETTING} Canonical w/ {BATCH_STR}"
                name = f"rew{r}_P_{PEARSON_SETTING}_C_{CANONICAL_SETTING}_{BATCH_STR}_traj"
                self.plot_traj(skills_to_plot, traj_to_plot, name, style_map=style_map, title=title, goal=self.agent.get_goal(), labels=labels_to_plot)

            # pick best skill
            best_skill_id = np.argsort(np.array(epic_losses[r]))[0]
            best_skills.append(skills[r][best_skill_id])
            best_skill_ids.append(best_skill_id)
        return

    def run_skills(self, skill_lst, tl_lst, **kwargs):
        out = dict(time_step=self.train_env.reset())
        final_out = dict()
        for sk, tl in zip(skill_lst, tl_lst):
            out = self.run_skill(sk, num_timesteps=tl, time_step=out['time_step'], **kwargs)
            for key, val in out.items():
                if key not in final_out:
                    final_out[key] = [val]
                else:
                    final_out[key].append(val)
        return final_out 

    def run_skill(self, skill, video=False, video_name=None, num_timesteps=None, time_step=None):
        step, episode, total_reward, env_reward = 0, 0, 0, 0
        ep_obs, ep_action = [], []
        meta = self.agent.init_meta(skill)

        if num_timesteps is None:
            time_step = self.train_env.reset()
            ep_obs.append(time_step.observation)

        self.video_recorder.init(self.train_env, enabled=video)
        while not time_step.last() and (num_timesteps is None or step < num_timesteps):
            prev_time_step = time_step
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=True)
            if "fetch" in self.cfg.domain:
                time_step = self.train_env.step(action, make_video=video)
            else:
                time_step = self.train_env.step(action)
            self.video_recorder.record(self.train_env)
            ep_obs.append(time_step.observation)
            ep_action.append(action)
            prev_obs, _ = self.agent.process_observation(torch.from_numpy(prev_time_step.observation).unsqueeze(0).to(self.cfg.device))
            curr_obs, _ = self.agent.process_observation(torch.from_numpy(time_step.observation).unsqueeze(0).to(self.cfg.device))
            total_reward += self.agent.get_extr_rew()(prev_obs, curr_obs, None)
            env_reward += time_step.reward
            step += 1
        if video:
            if video_name is None:
                video_name = f"{self.global_frame}.mp4"
            self.video_recorder.save(video_name)
        return dict(ep_obs=ep_obs, ep_action=ep_action, reward=total_reward, env_reward=env_reward, step=step, skill=meta['skill'], time_step=time_step)

    def plot_traj(self, skills, all_traj, name, style_map=None, title=None, goal=None, labels=None):
        if style_map is None:
            style_map = ['maroon', 'orangered', 'darkorange', 'gold', 'yellow', 
                        'lawngreen', 'aquamarine', 'deepskyblue', 'blue', 'darkblue',
                        'slateblue', 'rebeccapurple', 'indigo', 'blueviolet', 'mediumorchid',
                        'violet', 'magenta', 'orchid', 'deeppink', 'pink',
                        'gainsboro', 'silver', 'darkgray', 'dimgray', 'black']

        plt.xlim(self.xlim1, self.xlim2)
        plt.ylim(self.ylim1, self.ylim2)

        label = None
        if not self.plot_2d:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
        for i in range(len(all_traj)):
            if labels is not None:
                label = labels[i]
            traj_coord = np.array(all_traj[i])
            traj_coord, _ = self.agent.process_observation(torch.from_numpy(traj_coord).to(self.cfg.device))
            traj_coord = traj_coord.cpu().numpy()
            
            if self.plot_2d:
                plt.plot(traj_coord[:, 0], traj_coord[:, 1], c=style_map[i], label=label)
            else:
                plt.plot(traj_coord[:, 0], traj_coord[:, 1], traj_coord[:, 2], c=style_map[i], label=label)

        for g in goal:
            if g is None:
                pass
            elif self.plot_2d:
                plt.plot(g[0], g[1], marker="*", markersize="20")
            else:
                plt.plot(g[0], g[1], g[2], marker="*", markersize="20") 
        if title is not None:
            plt.title(title)
        if labels is not None:
            plt.legend(loc="upper right")
        plot_name = f"{self.plot_dir}/{name}.png"
        plt.savefig(plot_name)
        plt.clf()

        # write meta
        with open(f"{self.plot_dir}/meta_{name}.txt", "w") as f:
            for i in range(len(all_traj)):
                f.write(f"{style_map[i]}: {skills[i]}")
                f.write('\n')
        print(f"saved plot to {plot_name}")

    def scatter_plot(self, epic_losses, extr_rewards, title, plot_name=None, xlabel=None, ylabel=None):
        plt.scatter(epic_losses, extr_rewards, c="#28005f")
        plt.xlim(0,1)
        if xlabel is not None:
            plt.xlabel(xlabel)
        if ylabel is not None:
            plt.ylabel(ylabel)
        if title is not None:
            plt.title(title)
        plot_name = f"{self.plot_dir}/{plot_name}.png"
        plt.savefig(plot_name)
        plt.clf()        
        print(f"saved to {plot_name}")

    def compute_epic_loss(self, obs_np, skill, action, PEARSON_SETTING, CANONICAL_SETTING, same_batch=False):
        batch_size = obs_np.shape[0] - 1 # cuz need to account for obs, next_obs
        obs_dim = obs_np.shape[1]
        self.epic_samples = self.cfg.batch_size # for now, not sure if this is the best?
        obs, next_obs = torch.from_numpy(obs_np[:-1, :]).to(self.device), torch.from_numpy(obs_np[1:, :]).to(self.device)

        actions = torch.from_numpy(action[:-1, :]).to(self.device)
        skill_th = torch.from_numpy(skill).to(self.device)

        self.agent.pearson_setting, self.agent.canonical_setting = PEARSON_SETTING, CANONICAL_SETTING
        self.pearson_samples = batch_size

        loss = self.agent.compute_epic_loss_ft(skill_th, obs, next_obs, bounds=self.bounds)
        return loss

    def run(self):
        """
        We provide several interesting visualizations:
        - rew_epic_plot: A scatter plot of Extrinsic Reward vs. EPIC Loss. Also visualizes
                         trajectories with lowest and highest EPIC losses.
        - sequential_rew_epic_plot: As above, but with arbitrarily long-horizon sequential tasks.
        - epic_heat_plot: heat plot of EPIC loss across the discretized skill space 
                          (only works for 2d skills)
        """

        # for p, c in [("full_rand", "full_rand"), ("onpolicy", "full_rand"), ("onpolicy", "uniform")]:
        #     self.rew_epic_plot(p, c)

        # for p, c in [("prev_rollout", "full_rand"), ("prev_rollout", "gaussian_0.01"), ("prev_rollout", "gaussian_0.1")]:
        #     self.sequential_rew_epic_plot(p, c)

        # self.epic_heat_plot("full_rand", "full_rand")

@hydra.main(config_path='.', config_name='visualize_irm')
def main(cfg):
    from visualize_irm import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.run()


if __name__ == '__main__':
    main()
