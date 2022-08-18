import definitions as defs

### Environment ###
env = defs.classDefs.GridWorldEnv(grid_size = 21, dashRcoefInit = -0.5, dashLcoefInit = 1) # instantiate env
defs.check_env(env, warn = True) # check environment
env = defs.Monitor(env) # wrap environment for sb3 utilities
### Agent ###
agent = defs.PPO('MultiInputPolicy', env, verbose = 1)
agent.learn(total_timesteps = 100_000) # train agent
# Use trained agent
defs.functionDefs.watchTrained(env, agent, 200)