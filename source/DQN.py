from torch import device, cuda, Tensor
import torch.nn as nn

from torchinfo import summary


class BaseNetwork(nn.Module):
  def __init__(self, n_observations: int, n_actions: int) -> None:
    super(BaseNetwork, self).__init__()
    self.n_observations = n_observations
    self.n_actions = n_actions
    self.device = device('cuda:0' if cuda.is_available() else 'cpu')

  def forward(self, state: list) -> Tensor:
    pass

class DeepQNetworkV1(BaseNetwork):
  def __init__(self, n_observations: int, n_actions: int) -> None:
    super(DeepQNetworkV1, self).__init__(n_observations, n_actions)
    self.net = nn.Sequential(
      nn.Linear(n_observations, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, n_actions),
    )
    self.to(self.device)

  def forward(self, state: list) -> Tensor:
    return self.net(state)

if __name__ == '__main__':
  model = DeepQNetworkV1(16, 5)
  batch_size = 64
  summary(model, input_size=(batch_size, 16))
