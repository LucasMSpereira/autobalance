import numpy as np
import gym
import pygame
from gym import spaces
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed

# custom environment example
class GridWorldEnv(gym.Env):

  # render modes definition
  metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

  def __init__(self, grid_size: int = 5):
    super(GridWorldEnv, self).__init__()
    self.window_size = 512  # The size of the PyGame window
    self.size = grid_size
    
    # Observations are dictionaries with the agent's and the target's location.
    # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
    self.observation_space = spaces.Dict(
        {
            "agent": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, grid_size - 1, shape=(2,), dtype=int),
        }
    )

    # We have 4 actions, corresponding to "right", "up", "left", "down"
    self.action_space = spaces.Discrete(4)

    """
    The following dictionary maps abstract actions from `self.action_space` to 
    the direction we will walk in if that action is taken.
    I.e. 0 corresponds to "right", 1 to "up" etc.
    """
    self._action_to_direction = {
        0: np.array([1, 0]),
        1: np.array([0, 1]),
        2: np.array([-1, 0]),
        3: np.array([0, -1]),
    }

    """
    If human-rendering is used, `self.window` will be a reference
    to the window that we draw to. `self.clock` will be a clock that is used
    to ensure that the environment is rendered at the correct framerate in
    human-mode. They will remain `None` until human-mode is used for the
    first time.
    """
    self.window = None
    self.clock = None

    # turn current state of env. into observation.
    # will be used for 'step' and 'reset', but these
    # can be done separately
  
  def _get_obs(self):
    return {"agent": self._agent_location, "target": self._target_location}

  # auxiliary info from 'step' and 'reset' (manhattan distance)
  def _get_info(self):
    return {"distance": np.linalg.norm(self._agent_location - self._target_location, ord=1)}
  
  # reset env. in the beginning of an episode
  def reset(self, seed=None, return_info=False, options=None):
    # We need the following line to seed self.np_random
    # super().reset()

    # Choose the agent's location uniformly at random
    self._agent_location = np.random.randint(0, self.size - 1, size=2)

    # We will sample the target's location randomly until it does not coincide with the agent's location
    self._target_location = self._agent_location
    while np.array_equal(self._target_location, self._agent_location):
        self._target_location = np.random.randint(0, self.size - 1, size=2)

    observation = self._get_obs()
    info = self._get_info()
    return (observation, info) if return_info else observation

  # perform chosen action, update env. and
  # return (obs, reward, done, info)
  def step(self, action):
    # Map the action (element of {0,1,2,3}) to the direction we walk in
    direction = self._action_to_direction[action]
    # We use `np.clip` to make sure we don't leave the grid
    self._agent_location = np.clip(
        self._agent_location + direction, 0, self.size - 1
    )
    # An episode is done if the agent has reached the target
    done = np.array_equal(self._agent_location, self._target_location)
    reward = -np.linalg.norm(self._agent_location - self._target_location)/self.size
    observation = self._get_obs()
    info = self._get_info()

    return observation, reward, done, info

  def render(self, mode="human"):
    if self.window is None and mode == "human":
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
    if self.clock is None and mode == "human":
        self.clock = pygame.time.Clock()

    canvas = pygame.Surface((self.window_size, self.window_size))
    canvas.fill((155, 50, 155))
    pix_square_size = (
        self.window_size / self.size
    )  # The size of a single grid square in pixels

    # First we draw the target
    pygame.draw.rect(
        canvas,
        (255, 0, 100),
        pygame.Rect(
            pix_square_size * self._target_location,
            (pix_square_size, pix_square_size),
        ),
    )
    # Now we draw the agent
    pygame.draw.circle(
        canvas,
        (80, 80, 255),
        (self._agent_location + 0.5) * pix_square_size,
        pix_square_size / 3,
    )

    # Finally, add some gridlines
    for x in range(self.size + 1):
        pygame.draw.line(
            canvas,
            0,
            (0, pix_square_size * x),
            (self.window_size, pix_square_size * x),
            width=2,
        )
        pygame.draw.line(
            canvas,
            0,
            (pix_square_size * x, 0),
            (pix_square_size * x, self.window_size),
            width=2,
        )

    if mode == "human":
        # The following line copies our drawings from `canvas` to the visible window
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        # We need to ensure that human-rendering occurs at the predefined framerate.
        # The following line will automatically add a delay to keep the framerate stable.
        self.clock.tick(self.metadata["render_fps"])
    else:  # rgb_array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )

  # close display window in case of render mode = "human".
  # also "releases" used by the env.
  def close(self):
    if self.window is not None:
        pygame.display.quit()
        pygame.quit()



class brawlerTutEnv(gym.Env):
    
  metadata = {'render.modes': ['human']}

  def __init__(self):
    super(brawlerTutEnv, self).__init__()
    
    # Define action and observation space
    # They must be gym.spaces objects
    
    # left, right, jump, 2 types of attacks,
    # pick "warrior", pick "wizard"
    self.action_space = spaces.Discrete(7)
    
    # [selfHealth, selfX, selfY, enemyHealth, enemyX, enemyY, enemyAction, time]
    self.observation_space = spaces.Box(
      low = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
      high = np.array([100.0, 1000.0, 600.0, 100.0, 1000.0, 600.0, 5.0, 1e6]),
      dtype = np.float32
    )

  def step(self, action):
      observation = bt_main.engine(action)
      reward = (observation)
      return observation, reward, done, info
  def reset(self):
      ...
      return observation  # reward, done, info can't be included
  def render(self, mode='human'):
      ...
  def close (self):
      ...

