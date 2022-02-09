
"""Evaluate policies: render policy interactively, save videos, log episode return."""

import logging
import os
import os.path as osp
import time
import numpy as np
import pandas as pd
from typing import Any, Mapping, Optional
import torch

import gym
from sacred.observers import FileStorageObserver
from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common import base_class

import imitation.util.sacred as sacred_util
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.eval_policy import eval_policy_ex
from imitation.util import util, video_wrapper #reward_wrapper
from imitation.algorithms.bc import reconstruct_policy

import imageio

class InteractiveRender(VecEnvWrapper):
    """Render the wrapped environment(s) on screen."""

    def __init__(self, venv, fps):
        super().__init__(venv)
        self.render_fps = fps

    def reset(self):
        ob = self.venv.reset()
        self.venv.render()
        return ob

    def step_wait(self):
        ob = self.venv.step_wait()
        if self.render_fps > 0:
            time.sleep(1 / self.render_fps)
        self.venv.render()
        return ob


def video_wrapper_factory(log_dir: str, **kwargs):
    """Returns a function that wraps the environment in a video recorder."""

    def f(env: gym.Env, i: int) -> gym.Env:
        """Wraps `env` in a recorder saving videos to `{log_dir}/videos/{i}`."""
        directory = os.path.join(log_dir, "videos", str(i))
        return video_wrapper.VideoWrapper(env, directory=directory, **kwargs)

    return f

# some parameters setting
env_name = "NeedlePick-v0"  # environment to evaluate in
eval_n_timesteps = None  # Min timesteps to evaluate, optional.
eval_n_episodes = 2  # Num episodes to evaluate, optional.
num_vec = 1  # number of environments in parallel
parallel = False  # Use SubprocVecEnv (generally faster if num_vec>1)
max_episode_steps = None  # Set to positive int to limit episode horizons

videos = False  # save video files
video_kwargs = {}  # arguments to VideoWrapper
render = False  # render to screen
render_fps = -1  # -1 to render at full speed; 60 fps
log_root = os.path.join("output", "eval_policy")  # output directory

policy_type = "ppo"  # class to load policy, see imitation.policies.loader ppo
policy_path = (
    #"/home/zhaoogroup/code/tao_huang/SurRoL-imitation/GAIL/checkpoints/00200/gen_policy/",
    "/home/zhaoogroup/code/tao_huang/SurRoL-imitation-main/log/policy/BC_" + env_name + "_cnn_100.pt"
)  # serialized policy


train = False


reward_type = None  # Optional: override with reward of this type
reward_path = None  # Path of serialized reward to load

log_dir = os.path.join(
        log_root, env_name.replace("/", "_"), util.make_unique_timestamp()
    )

#####################################################################################
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logging.info("Logging to %s", log_dir)
sample_until = rollout.make_sample_until(eval_n_timesteps, eval_n_episodes)
post_wrappers = [video_wrapper_factory(log_dir, **video_kwargs)] if videos else None

results = np.zeros((1, 2))

venv = util.make_vec_env(env_name, n_envs=1, visual=True) 

try:
    #policy = serialize.load_policy(policy_type, policy_path, venv)
    policy = reconstruct_policy(policy_path)
    #trajs = rollout.generate_trajectories(policy, venv, sample_until)
    # print(rollout.rollout_stats(trajs))
    trajectories = rollout.generate_trajectories(policy, venv, sample_until,
                                                    deterministic_policy=True)
    
    folder = '/home/zhaoogroup/code/tao_huang/SurRoL-imitation-main'
    video_name = 'video_' + env_name + '_cnn_100.mp4'
    writer = imageio.get_writer(os.path.join(folder, video_name), fps=10)
    
    for traj in trajectories:
        for img in traj.obs:
            #print(img.transpose((1,2,0)))
            #print(np.clip(img.transpose((1,2,0))[:,:,3], 0, 255))
            writer.append_data(img.transpose((1,2,0)))
    writer.close()
    
    trained_stats = rollout.rollout_stats(trajectories)
    trained_ret_mean = trained_stats["return_mean"]
    trained_len_mean = trained_stats["len_mean"]
    #trained_suc_mean = trained_stats["success_mean"]
    results[0] = np.array(
        [trained_ret_mean, trained_len_mean])

    print('Validation: ret_mean: {}, len_mean:{}'.format(trained_ret_mean, trained_len_mean))

finally:
    venv.close()

# data_df = pd.DataFrame(results)
# data_df.columns = ['ret_mean', 'len_mean', 'suc_mean']

# #writer = pd.ExcelWriter('GAIL_002000.xlsx')
# writer = pd.ExcelWriter('BC_100.xlsx')
# data_df.to_excel(writer, 'page_1', float_format='%.5f')
# writer.save()