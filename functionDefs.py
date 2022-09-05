import numpy as np
import definitions as defs
import time
import gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# Function to be minimized through bayesian optimization (Ax package)
def balance(envParams):
  # create env with current set of parameters
  env = defs.classDefs.GridWorldEnv(dashLcoefInit = envParams["dashLcoef"], dashRcoefInit = envParams["dashRcoef"])
  agent = defs.PPO('MultiInputPolicy', env) # create agent
  agent.learn(total_timesteps = envParams['trainSteps']) # train agent for certain number of steps
  # Use trained agent for certain number of episodes
  chooseDashL, chooseDashR, episodes = useTrained(env, agent, envParams['evalEps'], showRender = True)
  return {"balance": abs(chooseDashL/(chooseDashL + chooseDashR + 1e-6) - 0.5), "episodes": episodes}

def evalAgent(env, steps, agent):
  obs = env.reset()
  for i in range(steps):
    print(f"Step {i}")
    action, _ = agent.predict(obs, deterministic = False)
    obs, _, done, _ = env.step(action)
    env.render()
    if done:
      obs = env.reset()

# Test specific configuration of dash coefficients
def explicitTest(trainSteps, leftCoef, rightCoef, evalEps, showRender):
  env = defs.classDefs.GridWorldEnv(dashLcoefInit = leftCoef, dashRcoefInit = rightCoef) # instantiate env with random parameters
  defs.check_env(env, warn = True); env = defs.Monitor(env) # check and wrap env
  agent = defs.PPO('MultiInputPolicy', env) # create agent
  agent.learn(total_timesteps = trainSteps) # train agent for certain number of steps
  dashL, dashR, eps = useTrained(env, agent, evalEps, showRender = showRender)
  print(f'dashL: {dashL}    dashR: {dashR}    episodes: {eps}')


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

# Ax optimization loop
def optLoop(numTrials, axClient, parameters):
  for i in range(numTrials): # optimization loop
      parameters, trial_index = axClient.get_next_trial() # query client for new trial
      axClient.complete_trial(trial_index = trial_index, raw_data = balance(parameters))

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

# Let trained agent play for certain amount of episodes
def useTrained(env, agent, numEps, showRender: bool = True, limitSteps: int = 1000):
  obs = env.reset() # Initially reset env
  # Initialize counters
  dashL = 0; dashR = 0; episodes = 0; numSteps = 0
  for eps in range(numEps): # Loop in desired amount of episodes
    while numSteps < limitSteps:
      # Policy determines action according to env state observed
      action, _ = agent.predict(obs, deterministic = True)
      obs, _, done, info = env.step(action) # Perform action and change env
      # Count number of times either dash were chosen
      if info['dash'] == "R":
        dashR += 1
      elif info['dash'] == "L":
        dashL += 1
      env.render() if showRender else False # Render step or not
      numSteps += 1
      if done: # Upon reaching target, reset env and count episodes
        obs = env.reset()
        episodes += 1
        break
  env.close()
  return dashL, dashR, episodes

# save
# model.save("./data/models/test") # save trained agent to zip
# load
# loaded_model = [type of agent (A2C, PPO, etc.)].load([path to zip], verbose=1)