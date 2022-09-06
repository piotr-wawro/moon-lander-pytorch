from pathlib import Path
import time

import gym
import PIL.Image

class Envirionment():
    """
    Proxy class for gym env.

    Attributes:
        env : Any
            Gym environment created by gym.make.
        observation_space : Any
            State that is returned by env.
        action_space : Any
            Actions that can be performed.
        timeout : int [s]
            After this time should_break changes to True.
        start_time : float
            Time when simulation started.
        should_break : bool
            Whether simulation should be stopped.
        score : float
            Cumulative score.
        done : bool
            Whether simulation is terminated.
    """

    def __init__(self, env_name: str, timeout: int = None, render: bool = False) -> None:
        if render:
            self.env = gym.make(env_name, new_step_api=True, render_mode='rgb_array')
        else:
            self.env = gym.make(env_name, new_step_api=True)

        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n

        self.timeout = timeout
        self.start_time = time.time()
        self.should_break = False

        self.score = 0
        self.done = False

    def reset(self):
        """Same as env.reset."""

        self.start_time = time.time()
        self.should_break = False
        self.score = 0
        self.done = False
        return self.env.reset()

    def step(self, action):
        """Same as env.step. Returns observation, reward, done."""

        if self.timeout and self.time_elapsed() > self.timeout:
            self.should_break = True

        observation, reward, done, *_ = self.env.step(action)
        self.score += reward
        self.done = done

        return observation, reward, done

    def time_elapsed(self) -> float:
        """Time in seconds since simulation start."""
        return time.time() - self.start_time

    def create_gif(self, path: Path) -> None:
        """
        Create gif from states of the environment since the last reset.
        
        Arguments:
            path : Path
                Path where gif will be saved.
            name : str
                Name of the file.
        """

        images = [PIL.Image.fromarray(arr) for arr in self.env.render()]
        images[0].save(fp=path, format='GIF', append_images=images[1:], save_all=True, duration=1000/60)
