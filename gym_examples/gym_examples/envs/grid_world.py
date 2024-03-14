import numpy as np
import pygame
import time

import gym
from gym import spaces
from gym.utils import seeding

class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=40):
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window

        self.seed()
        
        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, size - 1, shape=(2,), dtype=int),
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
            0: np.array([1, 0]), # right
            1: np.array([0, 1]), # up
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        
        self.keys_to_action={  # doctest: +SKIP
                "w": 1,
                "a": 2,
                "s": 3,
                "d": 0,
                }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
    # def _get_obs(self):
    #     return {"agent": self._agent_location, "target": self._target_location}
    def _get_obs(self):
        # Temporarily set render_mode to 'rgb_array' for this operation, if necessary
        original_render_mode = self.render_mode
        self.render_mode = 'rgb_array'
        
        # Generate the observation as an RGB array
        obs = self._render_frame()

        # Reset the render mode to its original state, if needed
        self.render_mode = original_render_mode
        
        return obs
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    def get_keys_to_action(self):
        return self.keys_to_action
    
    def reset(self, options=None):
        # We need the following line to seed self.np_random
        # super().reset()
        
        # # The minimum and maximum cooridinate are 0 and self.size - 1. But they're taken by the walls. So we can't use these positions.
        
        # Choose the agent's location uniformly at random
        self._agent_location = self.np_random.random_integers(1, self.size - 2, size=2)

        # We will sample the target's location randomly until it does not coincide with the agent's location
        
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.random_integers(
                1, self.size - 2, size=2
            )
            
            # print(self._target_location)
            # print(f"Size: [{self.size}, {self.size}]")

        self.current_reward = 0
        
        observation = self._get_obs()
        
        
        info = self._get_info() #  Not return info

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def step(self, action):
        # The `step()` method must return four values: obs, reward, done, info
        
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        
        # print(direction)
        
        next_location = self._agent_location + direction # planed next observation

        # print(f"Current position: {self._agent_location}")
        
        # Check for wall collisions
        if np.any(next_location <= 0) or np.any(next_location >= self.size - 1):
            # print(f"=====================> Collision!")
            
            # print(f"Before collision, the position is: {self._agent_location}")
            # print(f"Before collision, the planed next position is: {next_location}")
            # time.sleep(2)
            
            # Calculate inverse direction and ensure it's within grid limits
            inverse_direction = -direction * 3 #
            self._agent_location = np.clip(self._agent_location + inverse_direction, 0, self.size - 1)
            # print(f"After collision, the position is: {self._agent_location}")
            self.current_reward = 1  # Reward for hitting the wall and moving inversely
        else:
            # No collision, proceed as before
            self._agent_location = np.clip(next_location, 0, self.size - 1)
        
        # print(f"Next position: {self._agent_location}")
        
        terminated = np.array_equal(self._agent_location, self._target_location)
        self.current_reward += 10 if terminated else 0  # Existing reward logic
        
        # # We use `np.clip` to make sure we don't leave the grid
        # self._agent_location = np.clip(
        #     self._agent_location + direction, 0, self.size - 1
        # )
        # An episode is done iff the agent has reached the target
        
        observation = self._get_obs()
        info = self._get_info()
        done = terminated
        reward = self.current_reward
        # print(f"Reward of this step: {self.current_reward}")     
        
        if self.render_mode == "human":
            self._render_frame()

        
        return observation, reward, done, info
    
    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw walls
        wall_color = (0, 0, 0)  # Black
        wall_thickness = self.window_size / self.size  # Adjust thickness as needed. The thickness is just one grid.
        # print(f"wall_thickness: {wall_thickness}")
        pygame.draw.rect(canvas, wall_color, pygame.Rect(0, 0, self.window_size, wall_thickness))  # Top
        pygame.draw.rect(canvas, wall_color, pygame.Rect(0, self.window_size - wall_thickness, self.window_size, wall_thickness))  # Bottom
        pygame.draw.rect(canvas, wall_color, pygame.Rect(0, 0, wall_thickness, self.window_size))  # Left
        pygame.draw.rect(canvas, wall_color, pygame.Rect(self.window_size - wall_thickness, 0, wall_thickness, self.window_size))  # Right
        
        # # Finally, add some gridlines
        # for x in range(self.size + 1):
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (0, pix_square_size * x),
        #         (self.window_size, pix_square_size * x),
        #         width=3,
        #     )
        #     pygame.draw.line(
        #         canvas,
        #         0,
        #         (pix_square_size * x, 0),
        #         (pix_square_size * x, self.window_size),
        #         width=3,
        #     )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            returned_img = np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
            return returned_img
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    