"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""

import pathlib
import pickle
import tempfile
import gym

import stable_baselines3 as sb3
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    # CombinedExtractor,
    FlattenExtractor,
    MlpExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common import base_class
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

def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
    th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))
    # TODO(gleave): unify this with the saving logic in data_collect?
    # (Needs #43 to be merged before attempting.)
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
        trainer.venv_norm_obs,
    )

env_name = 'NeedlePick-v0'
root_dir = "/home/zhaoogroup/code/tao_huang/SurRoL-imitation-main"
rollout_path = os.path.join(root_dir, "data/train_rollout_" + env_name + "_100x_rgb.pkl") # train_rollout_400x
with open(rollout_path, "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

transitions = rollout.flatten_trajectories(trajectories)
# Train GAIL on expert data.
# GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
gail_logger = logger.configure(os.path.join(root_dir, 'log/tb'), ["tensorboard", "stdout"])

venv = util.make_vec_env(env_name, n_envs=1, visual=True)
print(venv.observation_space.shape)

# same parameters for saving policy
checkpoint_interval = 100

gail_logger = logger.configure(os.path.join(root_dir, 'log/gail/tb'), ["tensorboard", "stdout"])
start = time.time()
gail_trainer = gail.GAIL(
    venv=venv,
    demonstrations=transitions,
    demo_batch_size=64,
    gen_algo=sb3.PPO("CnnPolicy", venv, verbose=1, tensorboard_log="./output/GAIL_SurRoL_tensorboard/", n_steps=64),
    allow_variable_horizon=True,
    custom_logger=gail_logger,
    normalize_obs=False,
)

def gail_call_back(round_num):
    if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
        save(trainer=gail_trainer, save_path=os.path.join(root_dir, "GAIL", "checkpoints", f"{round_num:05d}"))

gail_trainer.train(total_timesteps=40960, callback=gail_call_back) #409600
end = time.time()
print("GAIL training time: ", end - start)



