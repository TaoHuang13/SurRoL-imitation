"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""

import pathlib
import pickle
import tempfile
import gym

import stable_baselines3 as sb3
from stable_baselines3.common import base_class
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    # CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)

from imitation.algorithms import adversarial, bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util
from imitation.policies import base, serialize
from imitation.algorithms.bc import reconstruct_policy

import os, glob
import numpy as np
import time
import torch as th
import imageio

env_name = 'NeedlePick-v0'

root_dir = "/home/zhaoogroup/code/tao_huang/SurRoL-imitation-main"
rollout_path = os.path.join(root_dir, "data/train_rollout_" + env_name + "_100x_rgbd_fixed.pkl") # train_rollout_400x
# Load pickled test demonstrations.
# with open("/home/curl/CUHK/Projects/RL/stable-baselines3-imitation/src/imitation/scripts/rollout_1x.pkl", "rb") as f:
with open(rollout_path, "rb") as f:
    trajectories = pickle.load(f)

folder = '/home/zhaoogroup/code/tao_huang/SurRoL-imitation-main'
video_name = 'demo_pick_rdgd.mp4'
writer = imageio.get_writer(os.path.join(folder, video_name), fps=10)
for traj in trajectories:
    for img in traj.obs:
        writer.append_data(img[:,:,3])
writer.close()

transitions = rollout.flatten_trajectories(trajectories)

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
#venv = util.make_vec_camsimenv_multiple("CamEnvSim-v0", render_mode='rgbd_array', n_envs=200, batch_index=0, root_dir=traj_dir, train=True) # 81
#venv = base_class.BaseAlgorithm._wrap_env(venv, monitor_wrapper=True)
#venv_eval   = util.make_vec_camsimenv_multiple("CamEnvSim-v0", render_mode='rgbd_array', n_envs=200, batch_index=0, root_dir=root_dir, train=False)

bc_logger = logger.configure(os.path.join(root_dir, 'log/tb'), ["tensorboard", "stdout"])

venv = util.make_vec_env(env_name, n_envs=1, visual=True)
#venv.observation_space = gym.spaces.Box(0, 255, shape=obs_shape, dtype='uint8')

start = time.time()
bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    custom_logger=bc_logger,
)
bc_trainer.train(n_epochs=100)
end = time.time()
print("BC training time: ", end - start)
bc_trainer.save_policy(os.path.join(root_dir, "log/policy/BC_" + env_name + "_cnn_100_rgbd_fixed.pt"))