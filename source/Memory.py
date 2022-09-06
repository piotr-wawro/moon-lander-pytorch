import random

from utils.SaveManager import Savable


class Memory(Savable):
    """
    A class responsible for storing experience replay. Replay is utilized during
    the training.

    Attributes:
        max_size : int = 1<<20
            Size of memory.
        memory : list
            Memory that stores tuples (state, action, new_state, reward, done).
    """

    def __init__(self, max_size: int = 1<<20) -> None:
        self.max_size = max_size
        self.memory = []
        self.rnd = random.Random(1)

    def save(self) -> dict:
        """
        Implementation of save method inherited from Savable class. Returns
        state of instance.
        """

        return {
            'max_size': self.max_size,
            'memory': self.memory
        }

    def restore(self, state) -> None:
        """Implementation of restore method inherited from Savable class."""

        self.max_size = state['max_size']
        self.memory = state['memory']

    def append(self, state, action, new_state, reward, done) -> None:
        """
        Creates tuple (state, action, new_state, reward, done) and appends it
        to memory. If memory is full, first element is removed.

        Arguments:
            state : Any
                State (observation) returned by gym environment step.
            action : Any
                Action chosen by DQN model.
            new_state : Any
                New State (observation) returned by gym environment step after
                preforming action.
            reward : Any
                Reward returned by gym environment step.
            done : Any
                Done (terminal state) returned by gym environment step.
        """

        if len(self.memory) == self.max_size:
            del self.memory[0]
        self.memory.append((state, action, new_state, reward, done))

    def free_memory(self, percent: float) -> None:
        """
        Free up memory.

        Arguments:
            percent : float [0,1]
                Percentage of minimum free space.
        """

        current = len(self.memory)/self.max_size
        target = 1-percent
        difference = current - target

        if difference > 0:
            to_delete = round(self.max_size*difference)
            del self.memory[:to_delete]

    def get_batch(self, size: int) -> list:
        """
        Returns list of random samples from memory.

        Arguments:
            size : int
                How many samples to return.
        """

        return self.rnd.sample(self.memory, size)

    def __len__(self) -> int:
        return len(self.memory)
