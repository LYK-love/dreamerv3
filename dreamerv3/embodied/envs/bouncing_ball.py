import embodied
import numpy as np
import cv2
import custom_envs.envs


class BouncingBallEnv(embodied.Env):

  def __init__(self, task, size=2, seed=None, velocity_scale = 1.0, ball_radius_ratio = 0.05, wall_thickness_ratio = 0.01, energy_loss_factor=0.9, apply_action=True, log=False):
    assert task in ('BouncingBallEnv')
    # print(f"apply_action: {apply_action}")
    self._env = custom_envs.envs.bouncing_ball.BouncingBallEnv(render_mode='rgb_array', size=size, seed=seed, velocity_scale = velocity_scale, ball_radius_ratio=ball_radius_ratio, wall_thickness_ratio = wall_thickness_ratio, energy_loss_factor=energy_loss_factor, apply_action=apply_action, log=log)
    self._done = True
    self._cv2 = cv2

  @property
  def obs_space(self):
    obs_space = self._env.observation_space
    # print(f"env.obs space shape: {obs_space.shape}")
    self._env.observation_space.shape = (64,64,3) # channel num?
    # s = self._env.observation_space.shape
    # print(f"self._env.observation_space.shape: {s}")
    spaces = {
        'image': embodied.Space(np.uint8, self._env.observation_space.shape),
        'reward': embodied.Space(np.uint8),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
    }
    return spaces

  @property
  def act_space(self):
    action = embodied.Space(np.float32, (2,), -1, 1)
    return {
        'action': action,
        'reset': embodied.Space(bool),
    }

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])
    
    reward = np.float32(reward)
    
    # print(f"info: {info}")
    
    # print(f"reward from env: {reward}")
    # if reward > 0:
    #   print("Booom ========================================")
    #   print(f"reward: {reward}")
    #   print("Booom ========================================")
      
    # print(f"converted reward: {reward}")
    
    terminated = False
    # terminated = np.array_equal(self._env._agent_location, self._env._target_location)
    is_terminal = terminated
     
    return self._obs(
        image, reward, info, is_terminal=is_terminal,
        is_last=self._done)

  def _obs(
      self, image, reward, info, is_terminal=False,
      is_first=False, is_last=False):
    image = self._cv2.resize(
            image, (64,64), interpolation=self._cv2.INTER_AREA)
    
    return dict(
        image=image,
        reward=reward,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal
    )

  def render(self):
    return self._env.render()
