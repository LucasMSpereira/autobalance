import numpy as np
import time
import pygame
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

def evalAgent(env, steps, agent):
  obs = env.reset()
  for i in range(steps):
    print(f"Step {i}")
    action, _ = agent.predict(obs, deterministic = False)
    obs, _, done, _ = env.step(action)
    env.render()
    if done:
      obs = env.reset()

# utility for multiprocessing (multiTrainEval)
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

# multiprocessing
def multiTrainEval(procNum, env_id, expNum, alg, trainStepNum, evalEnv, EvalEpsNum):

  reward_averages = []
  reward_std = []
  training_times = []
  total_procs = 0
  for n_procs in procNum: # each process
    print(n_procs)
    total_procs += n_procs
    print('Running for n_procs = {}'.format(n_procs))
    if n_procs == 1:
      # if there is only one process, there is no need to use multiprocessing
      train_env = DummyVecEnv([lambda: gym.make(env_id)])
    else:
      # Here we use the "fork" method for launching the processes, more information is available in the doc
      # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
      train_env = SubprocVecEnv([make_env(env_id, i + total_procs) for i in range(n_procs)], start_method = 'spawn')

    rewards = []
    times = []

    for experiment in range(expNum): # each experiment
      print(experiment)
      # it is recommended to run several experiments due to variability in results
      train_env.reset()
      model = alg('MlpPolicy', train_env, verbose = 0)
      start = time.time()
      model.learn(total_timesteps = trainStepNum)
      times.append(time.time() - start)
      mean_reward, _  = evaluate_policy(model, evalEnv, n_eval_episodes = EvalEpsNum)
      rewards.append(mean_reward)
    # Important: when using subprocess, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

def selfPlay(env, agt1, agt2, timeSteps):

  # Initialize variables needed for training
  agt1._setup_learn(timeSteps)
  agt2._setup_learn(timeSteps)

  obs1 = env.reset()
  obs2 = copy.copy(obs1) # both sides always see the same initial observation.

  done = False
  total_reward = 0

  while not done:

    act1 = agt1.predict(obs1)
    act2 = agt2.predict(obs2)
    
    # Collect experiences using the current policy and fill a "RolloutBuffer"
    continue_training = agt1.collect_rollouts()

    # Update policy using the currently gathered rollout buffer
    agt1.train()

    if continue_training is False:
      break

    obs1, reward, done, info = env.step(act1, act2) # extra argument
    obs2 = info['otherObs']

    total_reward += reward
    env.render()

  print("Agent 1's score:", total_reward)
  print("Agent 2's score:", -total_reward)

def watchTrained(env, agent, numEps):
  obs = env.reset()
  for _ in range(numEps):
    action, _states = agent.predict(obs)
    obs, rewards, dones, info = env.step(action)
    print(f'obs {obs}   rewards {rewards}')
    env.render()
    if dones:
      obs = env.reset()
    else:
      False
  env.close()


# save
# model.save("./data/models/test") # save trained agent to zip
# load
# loaded_model = [type of agent (A2C, PPO, etc.)].load([path to zip], verbose=1)