from gymnasium.utils.save_video import save_video
import gymnasium as gym
import numpy as np
import torch

from DQN import BaseNetwork
from Memory import Memory
from utils.SaveManager import Savable


class Agent(Savable):
  def __init__(self, environment: gym.Env, memory: Memory, policy_net: BaseNetwork, target_net: BaseNetwork,
               criterion, optimizer, epsilon_fun, gamma_fun, tau_fun, batch_fun) -> None:
    self.environment = environment
    self.memory = memory
    self.policy_net = policy_net
    self.target_net = target_net
    self.criterion = criterion
    self.optimizer = optimizer
    self.epsilon_fun = epsilon_fun
    self.gamma_fun = gamma_fun
    self.tau_fun = tau_fun
    self.batch_fun = batch_fun
    self.rnd = np.random.default_rng()

    self.episodes = 0
    self.steps = 0
    self.episode_length = []
    self.cumulative_reward = []
    self.average_loss = []

    self.target_net.load_state_dict(policy_net.state_dict())
    self.update_parameters()

  def update_parameters(self):
    self.epsilon = self.epsilon_fun(self.steps)
    self.gamma = self.gamma_fun(self.steps)
    self.tau = self.tau_fun(self.steps)
    self.batch = self.batch_fun(self.steps)
    self.memory.batch_size = round(self.batch)

  def save(self):
    return {
      'memory': self.memory.save(),
      'policy_net': self.policy_net.state_dict(),
      'target_net': self.target_net.state_dict(),
      'optimizer': self.optimizer.state_dict(),
      'episodes': self.episodes,
      'steps': self.steps,
      'episode_length': self.episode_length,
      'cumulative_reward': self.cumulative_reward,
      'average_loss': self.average_loss,
    }

  def restore(self, state):
    self.memory.restore(state['memory'])
    self.policy_net.load_state_dict(state['policy_net'])
    self.target_net.load_state_dict(state['target_net'])
    self.optimizer.load_state_dict(state['optimizer'])
    self.episodes = state['episodes']
    self.steps = state['steps']
    self.episode_length = state['episode_length']
    self.cumulative_reward = state['cumulative_reward']
    self.average_loss = state['average_loss']
    self.update_parameters()

  def store_transition(self, state, action, new_state, reward, done):
    self.memory.append(state, action, new_state, reward, done)

  def record_episode(self, env: gym.Env, path: str):
    state, info = env.reset()
    terminated = False
    truncated = False

    while not terminated and not truncated:
      action = self.choose_action(state, eval=True)
      new_state, reward, terminated, truncated, info = env.step(action)
      state = new_state

    save_video(
      env.render(),
      video_folder=path,
      episode_trigger=lambda x: True,
      fps=env.metadata["render_fps"],
      episode_index=self.episodes,
      step_starting_index=self.steps,
    )

  def episode(self):
    state, info = self.environment.reset()
    terminated = False
    truncated = False
    self.ep_length = 0
    self.cum_reward = 0
    self.avg_loss = []

    while not terminated and not truncated:
      action = self.choose_action(state)
      new_state, reward, terminated, truncated, info = self.environment.step(action)

      self.store_transition(state, action, new_state, reward, terminated or truncated)
      state = new_state

      self.optimize()
      self.soft_update()
  
      self.ep_length += 1
      self.cum_reward += reward

    self.episodes += 1
    self.episode_length.append(self.ep_length)
    self.cumulative_reward.append(self.cum_reward)
    self.average_loss.append(sum(self.avg_loss)/len(self.avg_loss) if len(self.avg_loss) > 0 else 0)
    return self.episode_length[-1], self.cumulative_reward[-1], self.average_loss[-1]

  def optimize(self):
    if len(self.memory) < self.memory.batch_size:
      return

    self.policy_net.train()

    batch = self.memory.sample()
    self.back_propagation(batch)

    self.policy_net.eval()

  def back_propagation(self, batch: int):
    state, action, new_state, reward, done = [np.vstack(l) for l in zip(*batch)]
    state = torch.tensor(state, dtype=torch.float32, device=self.policy_net.device)
    action = torch.tensor(action, dtype=torch.int64, device=self.policy_net.device)
    new_state = torch.tensor(new_state, dtype=torch.float32, device=self.policy_net.device)
    reward = torch.tensor(reward, dtype=torch.float32, device=self.policy_net.device)
    done = torch.tensor(done, dtype=torch.bool, device=self.policy_net.device).flatten()

    state_action_values = \
      self.policy_net.forward(state)\
      .gather(1, action)

    new_state_values = torch.zeros((
      self.memory.batch_size,
      self.policy_net.n_actions
    ), device=self.policy_net.device)
    with torch.no_grad():
      new_state_values[~done] = self.target_net.forward(new_state[~done])
    new_state_values = new_state_values.max(1, keepdim=True)[0]

    expected_state_action_values = reward + self.gamma * new_state_values
    loss = self.criterion(state_action_values, expected_state_action_values)
    self.avg_loss.append(loss.item())

    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

  def soft_update(self):
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
    self.target_net.load_state_dict(target_net_state_dict)

  def choose_action(self, state: list, eval: bool = False) -> int:
    state = torch.tensor(state, dtype=torch.float32, device=self.policy_net.device)

    if self.rnd.random() > self.epsilon or eval:
      with torch.no_grad():
        actions = self.policy_net.forward(state)
      chosen_action = actions.argmax().item()
    else:
      chosen_action = self.rnd.integers(self.policy_net.n_actions)

    if not eval:
      self.steps += 1
      self.update_parameters()

    return chosen_action
