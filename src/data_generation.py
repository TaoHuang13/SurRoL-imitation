"""
Data generation for the case of Psm Envs and demonstrations.
Refer to
https://github.com/openai/baselines/blob/master/baselines/her/experiment/data_generation/fetch_data_generation.py
"""
import os
import argparse
import gym
import time
import numpy as np
import imageio
from surrol.const import ROOT_DIR_PATH

from imitation.data import rollout, wrappers, types
from imitation.util.util import make_vec_env
import dataclasses
import logging

parser = argparse.ArgumentParser(description='generate demonstrations for imitation')
parser.add_argument('--env', type=str, required=True,
                    help='the environment to generate demonstrations')
parser.add_argument('--video', action='store_true',
                    help='whether or not to record video')
parser.add_argument('--steps', type=int,
                    help='how many steps allowed to run')
args = parser.parse_args()

rollout_save_path = "/home/zhaoogroup/code/tao_huang/SurRoL-imitation-main/data/train_rollout_" + args.env + "_100x_rgb_fixed_test.pkl"

def main():
    env = gym.make(args.env, render_mode='human')  # 'human'     
    trajectories_accum = rollout.TrajectoryAccumulator()
    num_itr = 100 if not args.video else 1
    
    print("Reset!")
    init_time = time.time()

    if args.steps is None:
        args.steps = env._max_episode_steps

    print()
    goToGoal(env, trajectories_accum, num_itr)

    used_time = time.time() - init_time
    print("Saved data at:", rollout_save_path)
    print("Time used: {:.1f}m, {:.1f}s\n".format(used_time // 60, used_time % 60))
    env.close()


def goToGoal(env, trajectories_accum, num_itr):
    trajectories = []

    obs = env.reset()
    for i in range(num_itr):
        time_step = 0  # count the total number of time steps
        episode_init_time = time.time()
        success = False

        img = env.render('rgb_array')
        if i == 0:
            trajectories_accum.add_step(dict(obs=img), 0)   # totally one more obs than act

        while time_step < min(env._max_episode_steps, args.steps):
            action = env.get_oracle_action(obs)
            obs, reward, done, info = env.step(action)
            done = info['is_success']

            if done:
                obs = env.reset()
            img = env.render('rgb_array')
            
            new_trajs = trajectories_accum.add_steps_and_auto_finish(
                np.expand_dims(action,axis=0), 
                np.expand_dims(img,axis=0),
                np.expand_dims(reward,axis=0),
                np.expand_dims(done,axis=0), 
                np.expand_dims(info,axis=0),
                is_surrol=True,
            )

            trajectories.extend(new_trajs)
            
            # print(f" -> obs: {obs}, reward: {reward}, done: {done}, info: {info}.")
            time_step += 1

            if isinstance(obs, dict) and info['is_success'] > 0 and not success:
                print("Timesteps to finish:", time_step)
                success = True
            
            if done:
                break
        print(len(trajectories))
        print("Episode time used: {:.2f}s\n".format(time.time() - episode_init_time))

    #np.random.shuffle(trajectories)
    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        exp_obs = (n_steps + 1,) + trajectory.obs.shape[1:]
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + env.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    unwrap = False
    exclude_infos = True
    verbose = 1
    if unwrap:
        trajectories = [rollout.unwrap_traj(traj) for traj in trajectories]
    if exclude_infos:
        trajectories = [dataclasses.replace(traj, infos=None) for traj in trajectories]
    if verbose:
        stats = rollout.rollout_stats(trajectories)
        logging.info(f"Rollout stats: {stats}")

    types.save(rollout_save_path, trajectories)

if __name__ == "__main__":
    main()
