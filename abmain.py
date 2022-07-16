import defs # definitions
import stable_baselines3
import gym
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.monitor import Monitor

### Environment ###
env = defs.GridWorldEnv(grid_size = 20) # instantiate env
check_env(env, warn=True) # check environment
env = Monitor(env) # wrap environment for sb3 utilities

### Agent ###
model = A2C('MultiInputPolicy', env, verbose=1).learn(10000) # train agent
# save
# model.save("./data/models/test") # save trained agent to zip
# load
# loaded_model = [type of agent (A2C, PPO, etc.)].load([path to zip], verbose=1)
obs = env.reset()
n_steps = 20
# use trained agent
for step in range(n_steps):
  action, _ = model.predict(obs)
  print("Step {}".format(step + 1))
  print("Action: ", action)
  obs, reward, done, info = env.step(action)
  print('obs=', obs, 'reward=', reward, 'done=', done)
  env.render(mode='human')
  if done:
    # Note that the VecEnv resets automatically
    # when a done signal is encountered
    print("Goal reached!", "reward=", reward)
    break