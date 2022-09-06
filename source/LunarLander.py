# %%
from pathlib import Path

import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.SaveManager import SaveManager
from Environment import Envirionment
from DQN import DeepQNetwork
from Agent import Agent
from Memory import Memory

# %%
GYM = 'LunarLander-v2'

environment = Envirionment(GYM, timeout=2)
model = DeepQNetwork(input_dims=8, fc1_dims=256, fc2_dims=256, n_actions=4)
memory = Memory(1<<18)
agent = Agent(environment, model, memory, nn.MSELoss(), optim.RMSprop(model.parameters()), epsilon_fun=lambda x: 111111/(x+111111))

# %%
save_manager = SaveManager(Path('dumps', GYM), agent, min_saves=1, max_saves=5, max_size=512, time_interval=300)
save_manager.load()
agent.model.eval()

# %%
FRAMES = 15000000
with tqdm(total=FRAMES, unit='frames', desc="watched frames", unit_scale=True) as pbar:
    while agent.frames < FRAMES:
        pbar.n = agent.frames
        pbar.display()

        agent.gather_experience(fill=0.2)
        agent.learn(loss=10)

        save_manager.save()

        # if agent.epoch%5==0:
        # agent.create_gif(Envirionment(GYM, timeout=30, render=True), Path('videos', f"{GYM}-{agent.epoch}.gif"))

save_manager.save(force=True)

# %%
fig, ax = plt.subplots(2, 1)

ax[0].plot(agent.avg_score[0], agent.avg_score[1])
ax[0].set(ylabel='score')

ax[1].plot(agent.avg_loss[0], agent.avg_loss[1])
ax[1].set(ylabel='loss', xlabel='epoch')

plt.show()
fig.savefig(Path('dumps', GYM))

# %%
