import glob
import matplotlib.pyplot as plt 
import numpy as np
import os
import pandas as pd
plt.style.use('seaborn')
import math

points = 100
smooth = True
if __name__ == '__main__':
    logdir = ("LOGDIR") # just modify the logdir to specify what to plot
    task_csv_paths = glob.glob(logdir)
    # make figures for each task
    for task_csv_path in task_csv_paths: 
        experiment_csv_paths = glob.glob(task_csv_path + "/*")
        # make a figure for each experiment
        for experiment_csv_path in experiment_csv_paths:
            tokens = str(experiment_csv_path).split('/')
            experiment_name = tokens[-1]
            task_name = tokens[-2][16:] 
            if 'walker' in task_name:
                ep_len = 100
            elif 'jaco' in task_name:
                ep_len = 125
            elif 'quadruped' in task_name:
                ep_len = 80
            zid_csv_paths = glob.glob(experiment_csv_path + "/*")
            colors = ['blue', 'black', 'gray', 'green', 'teal', 'purple', 'indigo']
            i=0
            zero_shot_means = {}
            zero_shot_stds = {}
            # make a plot for each zid method
            irm_paths = []
            non_irm_paths = []
            for path in zid_csv_paths:
                if "irm" in path:
                    irm_paths.append(path)
                else:
                    non_irm_paths.append(path)
            zid_csv_paths = irm_paths + sorted(non_irm_paths)
            for zid_csv_path in zid_csv_paths:
                zid = str(zid_csv_path).split('/')[-1]
                if zid == "env_rollout": 
                    offset = ep_len * 10
                elif zid == "env_rollout_cem":
                    offset = ep_len * 1000 * 5
                elif zid == "grid_search":
                    offset = ep_len * 10
                else:
                    offset = 0
                offset = min(offset, 99000)
                seed_csv_paths = glob.glob(zid_csv_path + "/*/eval.csv")
                num_seeds = len(seed_csv_paths)
                seed_data = []
                seed_rewards = []
                # use seeds to provide error bars
                for seed_csv_path in seed_csv_paths:
                    seed_csv_data = pd.read_csv(seed_csv_path).to_dict()
                    # get rid of pandas row index
                    for entry in seed_csv_data:
                        seed_csv_data[entry] = [seed_csv_data[entry][value] for value in seed_csv_data[entry]]
                    if len(seed_csv_data['episode_reward']) < points:
                        num_seeds -= 1
                        continue
                    rews = seed_csv_data['episode_reward'][:points]
                    smoothed_rews = rews 
                    if smooth:
                        smoothed_rews = []
                        for point in rews:
                            if len(smoothed_rews) == 0:
                                smoothed_rews.append(point)
                            else:
                                last = smoothed_rews[-1]
                                last = last * 0.9 + point * 0.1
                                smoothed_rews.append(last)
                    seed_data.append(seed_csv_data)
                    seed_rewards.append(smoothed_rews)
                    step = seed_data[0]['step'][:points]
                seed_rewards = np.array(seed_rewards) 
                seed_rewards = [r for r in seed_rewards]
                seed_mean_reward = np.mean(seed_rewards, axis=0)
                zero_shot_means[zid.replace("_", " ")] = seed_mean_reward[0]
                seed_ste_reward = np.std(seed_rewards, axis=0) / np.sqrt(num_seeds)
                zero_shot_stds[zid.replace("_", " ")] = seed_ste_reward[0]
                low = seed_mean_reward - seed_ste_reward
                high = seed_mean_reward + seed_ste_reward
                offset_step = [s + offset for s in step]
                offset_endpoint = 100 - math.ceil(offset / 1000)
                plt.fill_between(offset_step[:offset_endpoint], low[:offset_endpoint] , high[:offset_endpoint], alpha=0.1, color=colors[i % len(colors)])
                plt.plot(offset_step[:offset_endpoint], seed_mean_reward[:offset_endpoint], label=zid.replace("_", " "), color=colors[i % len(colors)])
                i += 1

            # print(task_name)
            # print(zero_shot_means)
            # print(zero_shot_stds)
            # print("\n")

            plt.xlabel("environment steps")
            plt.ylabel("average return")
            plt.title(task_name.replace("_", " "))
            plt.legend(loc="lower right")
            if task_name == "quadruped_stand":
                plt.legend(loc="upper right")
            plot_dir = 'iclr_plots/finetunenorewardfree'
            if not os.path.exists(plot_dir):
                os.mkdir(plot_dir)
            save_path = plot_dir + '/' + task_name + '_' + experiment_name + '.png'
            plt.savefig(save_path)   
            plt.clf()

            values = zero_shot_means.values() 
            keys = zero_shot_means.keys()
            keys = [k.replace(" ", "\n") for k in keys]
            plt.bar(range(len(values)), values, align='center')
            plt.style.use('seaborn')
            plt.errorbar(range(len(values)), values, yerr=zero_shot_stds.values(), color='black', fmt="+")
            plt.title(task_name.replace("_", " "))
            plt.ylabel("zero-shot average return")
            plt.xticks(range(len(values)), keys)
            save_path = plot_dir + '/' + task_name + '_' + experiment_name + '_zeroshot.png'
            plt.savefig(save_path)
            plt.clf()