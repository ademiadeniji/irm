# Intrinsic Reward Matching (IRM)

This codebase contains implementation, visualizations, and analysis of IRM from Skill-Based Reinforcement Learning with Intrinsic Reward Matching (Adeniji and Xie et al 2022). IRM leverages the skill discriminator from unsupervised RL pretraining to perform environment-interaction-free skill sequencing for unseen downstream tasks. 

<img src="imgs/method.png" alt="irm method figure" title="irm method figure">

This codebase is built on top of the [Contrastive Intrinsic Control (CIC) codebase](https://github.com/rll-research/cic). 

Before running IRM, you must first pretrain an agent with the unsupervised RL method of your choice.

To pretrain an agent `dads` or `cic`, run the following command:

```sh
python pretrain.py agent=AGENT domain=DOMAIN experiment_folder=YOUR_EXP_FOLDER experiment_name=YOUR_EXP_NAME 
```

To finetune your pretrained agent, run the following command. Make sure to specify the directory of your saved snapshots with `YOUR_EXP_NAME`.

```sh
python finetune.py agent=AGENT experiment=YOUR_EXP_NAME task=TASK extr_reward=REWARD restore_snapshot_ts=2000000 restore_snapshot_dir=PATH_TO_PRETRAINED_MODEL
```

In addition, we include a visualization script. You can use this script to see detailed insights into the IRM skill selection process. 

```sh
python visualize_irm.py agent=AGENT experiment=YOUR_EXP_NAME domain=DOMAIN restore_snapshot_ts=2000000 restore_snapshot_dir=PATH_TO_PRETRAINED_MODEL
```
<img src="imgs/correlation.png" alt="correlation analysis" title="correlation analysis">

For sequential task finetuning (or IRM visualizations), add the flags `extr_reward_seq=[REW1,REW2,REW3]` and `extr_reward=REW1`.

## Requirements 
We assume you have access to a GPU that can run CUDA 10.2 and CUDNN 8. Then, the simplest way to install all required dependencies is to create an anaconda environment by running
```sh
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with
```sh
conda activate irm
```

Note that we use a [custom implementation](https://github.com/amberxie88/gym) of OpenAI Gym's Fetch environment.

## Available Domains
We work on the following domains + tasks:
| Domain | Tasks | Reduced State |
|---|---|---|
| `fetch_reach` | `goal_0.5_0.5_0.5`, `goal_1_1.2_1`  | `fetch_reach_xyz` |
| `fetch_push` | `goal_barrier1`, `goal_barrier2`, `goal_barrier3` | `fetch_push_xy` |
| `fetch_barrier` | `goal_barrier1`, `goal_barrier2`, `goal_barrier3` | `fetch_push_xy` |
| `walker` | `stand`, `walk`, `run`, `flip` | `walker_delta_xyz` |
| `quadruped` | `walk`, `run`, `stand`, `jump` | `quadruped_velocity` |
| `jaco` | `reach_top_left`, `reach_top_right`, `reach_bottom_left`, `reach_bottom_right` | `jaco_xyz` |
| `plane` | `goal_top_right`, `goal_top_left` | `states`

<p float="left" align="middle">
<!--   <img src="imgs/heat.png" alt="heatmap visualization" title="heatmap visualization" width=500> -->
  <img src="imgs/fetch.png" alt="Fetch domain" title="Fetch domain" width=500>
</p>

### Monitoring
Logs are stored in the `exp_local` folder. To launch tensorboard run:
```sh
tensorboard --logdir exp_local
```

You may also enable logging to wandb and view logs there.

### Citation
If you use this code in your own research, please consider citing:
```sh
todo
```

### todos
- formatting (at end)
- add pretrained agent checkpoints for the fetch tasks so researchers can replicate without pretraining
- citation
