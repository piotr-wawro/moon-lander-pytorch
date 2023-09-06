from collections import deque
from random import Random

from utils.SaveManager import Savable


class Memory(Savable):
  def __init__(self, max_size: int = 100_000, batch_size: int = 64) -> None:
    self.max_size = max_size
    self.batch_size = batch_size
    self.memory = deque([], maxlen=max_size)
    self.rnd = Random(1)

  def save(self) -> dict:
    return {
      'max_size': self.max_size,
      'batch_size': self.batch_size,
      'memory': self.memory,
    }

  def restore(self, state) -> None:
    self.max_size = state['max_size']
    self.batch_size = state['batch_size']
    self.memory = state['memory']

  def append(self, state, action, new_state, reward, done) -> None:
    self.memory.append((state, action, new_state, reward, done))

  def sample(self) -> list:
    return self.rnd.sample(self.memory, self.batch_size)

  def __len__(self) -> int:
    return len(self.memory)
