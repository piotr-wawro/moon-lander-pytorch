from pathlib import Path
import random
from copy import deepcopy

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from Memory import Memory
from utils.SaveManager import Savable


class Agent(Savable):
    """
    Attributes:
        environment : Environment
        model : nn.Module
        memory : Memory
        loss : Any
        optimize : Any
        epsilon_fun : lambda x
        epsilon : float
        gamma : float = 0.99
        batches : int = None
        batch_size : int = 64
    """

    def __init__(self, environment, model: nn.Module, memory: Memory, loss, optimizer, epsilon_fun,
                 gamma: float = 0.99, batches: int = None) -> None:
        self.environment = environment
        self.model = model
        self.memory = memory

        self.model_copy = deepcopy(model)

        self.epoch = 0
        self.frames = 0
        self.avg_score = [[],[]]
        self.avg_loss = [[],[]]

        self.loss = loss
        self.optimizer = optimizer

        self.rnd = random.Random(1)
        self.epsilon_fun = epsilon_fun
        self.epsilon = self.epsilon_fun(self.frames)
        self.gamma = gamma

        self.batch_size = round(self.memory.max_size*0.001)
        self.batches = len(self.memory)//self.batch_size

        self.action_space = [i for i in range(model.n_actions)]

    def save(self):
        """
        Implementation of save method inherited from Savable class. Returns
        state of instance.
        """

        return {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'memory': self.memory.save(),
            'epoch': self.epoch,
            'frames': self.frames,
            'avg_score': self.avg_score,
            'avg_loss': self.avg_loss,
            'epsilon': self.epsilon,
        }

    def restore(self, state):
        """Implementation of restore method inherited from Savable class."""

        self.model.load_state_dict(state['model'])
        self.model_copy = deepcopy(self.model)
        self.optimizer.load_state_dict(state['optimizer'])
        self.memory.restore(state['memory'])
        self.epoch = state['epoch']
        self.frames = state['frames']
        self.avg_score = state['avg_score']
        self.avg_loss = state['avg_loss']
        self.epsilon = state['epsilon']

    def store_transition(self, state, action, new_state, reward, done):
        """Store state in memory."""
        self.memory.append(state, action, new_state, reward, done)

    def choose_action(self, obesrvation: list) -> int:
        """Choose action based on observation."""

        if self.rnd.random() > self.epsilon:
            state = torch.tensor(obesrvation, dtype=torch.float).to(self.model.device)
            actions = self.model.forward(state)
            # weights = torch.softmax(actions, dim=0).tolist()
            # action = self.rnd.choices(self.action_space, weights, k=1)[0]
            action = torch.argmax(actions).item()
        else:
            action =  self.rnd.choice(self.action_space)

        return action

    def gather_experience(self, fill: float = 0.1, frameskip: int = 2):
        """
        Play in environment to fill up memory.

        Arguments:
            fill : float = 0.1 [0,1]
                Percentage value how much memory to fill.
            frames : int = 2
                How many frames will perform the same action.
        """

        frames_to_store = self.memory.max_size*fill
        self.memory.free_memory(fill)
        stored_frames = 0
        score_arr = []

        with tqdm(total=frames_to_store, unit='frames', desc="stored frames", unit_scale=True) as pbar:
            while stored_frames < frames_to_store:
                observation = self.environment.reset()

                while not self.environment.done and not self.environment.should_break:
                    action = self.choose_action(observation)

                    reward_sum = 0
                    for i in range(frameskip):
                        new_observation, reward, done = self.environment.step(action)
                        reward_sum += reward

                    self.store_transition(observation, action, new_observation, reward_sum, done)
                    stored_frames += 1

                    observation = new_observation

                pbar.set_postfix({
                    "simulation_time": f"{self.environment.time_elapsed():_>6.2f}s",
                    "score": f"{self.environment.score:_>+8.2f}",
                    "memory_used": f"{len(self.memory.memory)/self.memory.max_size*100:_>6.2f}%",
                }, refresh=False)
                pbar.n = stored_frames
                pbar.display()

                score_arr.append(self.environment.score)

            self.avg_score[0].append(self.epoch)
            self.avg_score[1].append(sum(score_arr)/len(score_arr))
            pbar.set_postfix({
                "epoch": f"{self.epoch}",
                "avg_score": f"{self.avg_score[1][-1]:_>+8.2f}",
            })

    def create_gif(self, env, path: Path):
        observation = env.reset()

        with tqdm(unit='frames') as pbar:
            while not env.done and not env.should_break:
                action = self.choose_action(observation)

                for i in range(2):
                    new_observation, *_ = env.step(action)

                observation = new_observation

                pbar.set_postfix({
                    "score": f"{env.score:_>+8.2f}",
                }, refresh=False)
                pbar.update()

        env.create_gif(path)

    def learn(self, epochs: int = None, loss: float = None) -> float:
        self.batches = len(self.memory)//self.batch_size

        if len(self.memory)//self.batch_size < 1:
            return

        if epochs:
            for i in tqdm(range(epochs), desc="epoch", unit="epochs"):
                self._next_epoch()
        elif loss:
            with tqdm(desc="epoch", unit="epochs") as pbar:
                model_update = 0

                while True:
                    self._next_epoch()
                    pbar.display()

                    if len(self.avg_loss[1]) > 2 and np.average(np.abs(np.diff(self.avg_loss[1][-3:]))) < loss:
                        self.model_copy = deepcopy(self.model)
                        model_update += 1
                    else:
                        model_update = 0

                    if model_update > 2:
                        break

    def _next_epoch(self):
        epoch_loss = []
        self.model.train()

        with tqdm(total=self.batches, desc="learning", unit="batches") as pbar:
            for i in range(self.batches):
                self.frames += self.batch_size
                batch = self.memory.get_batch(self.batch_size)
                batch = list(zip(*batch))

                state = torch.tensor(np.array(batch[0]), dtype=torch.float).to(self.model.device)
                action = batch[1]
                new_state = torch.tensor(np.array(batch[2]), dtype=torch.float).to(self.model.device)
                reward = torch.tensor(np.array(batch[3]), dtype=torch.float).to(self.model.device)
                done = torch.tensor(np.array(batch[4]), dtype=torch.bool).to(self.model.device)

                next = self.model_copy.forward(new_state)
                next[done] = 0.0
                target = reward + self.gamma * torch.max(next, dim=1)[0]

                q_eval = self.model.forward(state)[np.arange(self.batch_size), action]
                loss = self.loss(target, q_eval)
                epoch_loss.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.update()

            self.epsilon = self.epsilon_fun(self.frames)
            self.epoch += 1

            self.avg_loss[0].append(self.epoch)
            self.avg_loss[1].append(sum(epoch_loss)/len(epoch_loss))

            pbar.set_postfix({
                "epoch": f"{self.epoch}",
                "avg_loss": f"{self.avg_loss[1][-1]:_>8.2f}",
            })

        self.model.eval()
