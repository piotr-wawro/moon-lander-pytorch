# %%
from pathlib import Path

from tqdm import tqdm
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim

from Agent import Agent
from DQN import DeepQNetworkV1
from Memory import Memory
from utils.SaveManager import SaveManager

# %%
environment = gym.make("LunarLander-v2", max_episode_steps=1000, enable_wind = True, wind_power = 5.0, turbulence_power = 1.0)
record_env = gym.make("LunarLander-v2", max_episode_steps=1000, render_mode='rgb_array_list', enable_wind = True, wind_power = 5.0, turbulence_power = 1.0)
memory = Memory(max_size=100_000, batch_size=64)
policy_net = DeepQNetworkV1(n_observations=8, n_actions=4)
target_net = DeepQNetworkV1(n_observations=8, n_actions=4)
criterion = nn.SmoothL1Loss()
optimizer = optim.RMSprop(policy_net.parameters(), lr=0.001)
epsilon_fun = lambda x: 0.05 + (0.9 - 0.05) * np.exp(-x / 10000)
gamma_fun = lambda x: 0.98
tau_fun = lambda x: 0.001
batch_fun = lambda x: 64
agent = Agent(environment, memory, policy_net, target_net, criterion,
              optimizer, epsilon_fun, gamma_fun, tau_fun, batch_fun)

# %%
save_manager = SaveManager(Path('dumps'), agent, min_saves=1, max_saves=5, max_mb_size=512, time_interval=300)
save_manager.load('2023-09-06 20-00-29.tar')
agent.policy_net.eval()
agent.target_net.eval()

# %%
EPISODES = 200
with tqdm(total=EPISODES, desc="Episodes", unit='ep', unit_scale=True) as pbar:
  for i in range(EPISODES):
    length, reward, loss = agent.episode()
    pbar.set_postfix({
      'epsilon': f'{agent.epsilon:_>3.2f}',
      'steps': f'{agent.steps:_>6}',
      'ep length': f'{length:_>4}',
      'reward': f'{reward:_>+7.2f}',
      'loss': f'{loss:_>8.2e}',
    })
    pbar.update()

    if (i+1) % 100 == 0:
      save_manager.save()
      agent.record_episode(record_env, 'videos')

save_manager.save(force=True)

# %%
fig, ax = plt.subplots(3, 1)

ax[0].plot(range(len(agent.cumulative_reward)), agent.cumulative_reward)
ax[0].set(ylabel='reward')

ax[1].plot(range(len(agent.average_loss)), agent.average_loss)
ax[1].set(ylabel='loss')

ax[2].plot(range(len(agent.episode_length)), agent.episode_length)
ax[2].set(ylabel='episode_length', xlabel='episode')

plt.show()
# fig.savefig(Path('fig'))
