"""
File to implement the snake environment
"""
import numpy as np


class Environment:
    """
    Mother class for Environment, implement the following methods:
    - reset : None -> None | reset the environment
    - act : Action -> state, action, next_state, reward, done
    """

    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self):
        pass


class Snake(Environment):
    """
    Implementation of the Snake Environment

    State : numpy array RxC, -1 = Apple, 0 = Nothing, 1 = Snake Body, 2 = Snake head
    """

    def __init__(self, dimension, init_size, reward_loss, reward_apple):
        self.dimension = dimension
        self.size = init_size
        self.reward_loss = reward_loss
        self.reward_apple = reward_apple

        error_msg = "The initial size should be smaller than the number of rows"
        assert self.size <= self.dimension, error_msg

        # Initial the snake vertically in the middle
        self.board = np.zeros((self.dimension, self.dimension))
        self.board[int(self.dimension / 2) - int(self.size / 2), int(self.dimension / 2)] = 2
        self.board[
        (int(self.dimension / 2) - int(self.size / 2) + 1):(int(self.dimension / 2) + self.size - int(self.size / 2)),
        int(self.dimension / 2)] = 1

        # List of coordinates where the snake has turned
        self.elbows = [
            np.array([int(self.dimension / 2) + self.size - int(self.size / 2) - 1, int(self.dimension / 2)]),
            np.array([int(self.dimension / 2) - int(self.size / 2), int(self.dimension / 2)])]

        self._sample_apple()

        self.frame = np.zeros((1, self.dimension, self.dimension, 3))

        self.action_to_np = {0: np.array([-1, 0]), 1: np.array([0, 1]), 2: np.array([1, 0]), 3: np.array([0, -1])}

    def reset(self):
        # Initial the snake vertically in the middle
        self.board = np.zeros((self.dimension, self.dimension))
        self.board[int(self.dimension / 2) - int(self.size / 2), int(self.dimension / 2)] = 2
        self.board[
        (int(self.dimension / 2) - int(self.size / 2) + 1):(int(self.dimension / 2) + self.size - int(self.size / 2)),
        int(self.dimension / 2)] = 1
        self._sample_apple()

        self.frame = np.zeros((1, self.dimension, self.dimension, 3))

    def act(self, action):
        """
        Return state, action, next_state, reward, done
        state : Current state
        action : action taken
        next_state : arriving state
        reward : reward from action
        done : Boolean if game finished
        """
        ## 0 : Top, 1 : Right, 2 : Bottom, 3 : Left
        state = np.copy(self.board)

        self.board[self.elbows[-1][0], self.elbows[-1][1]] = 1

        if np.all(self.action_to_np[action] == np.sign(self.elbows[-1] - self.elbows[-2])):
            self.elbows[-1] += self.action_to_np[action]
        else:
            self.elbows.append(self.elbows[-1] + self.action_to_np[action])

        ## Four possibilities : Either wall, nothing apple or snake
        if np.any(self.elbows[-1] >= self.dimension) or np.any(self.elbows[-1] < 0):
            return state, action, [], self.reward_loss, True

        if self.board[self.elbows[-1][0], self.elbows[-1][1]] == -1:
            self.board[self.elbows[-1][0], self.elbows[-1][1]] = 2
            self.size += 1
            if self.size < self.dimension ** 2:
                self._sample_apple()
                return state, action, self.board, self.reward_apple, False
            else:
                return state, action, self.board, self.reward_apple, True

        self._contract()

        if self.board[self.elbows[-1][0], self.elbows[-1][1]] == 0:
            self.board[self.elbows[-1][0], self.elbows[-1][1]] = 2
            return state, action, self.board, 0, False

        if self.board[self.elbows[-1][0], self.elbows[-1][1]] >= 1:
            return state, action, [], self.reward_loss, True

    def _sample_apple(self):
        nb_possibility = self.dimension ** 2 - self.size
        ap = np.random.randint(nb_possibility)
        i = 0
        while ap > 0 or self.board[i // self.dimension, i % self.dimension] >= 1:
            if self.board[i // self.dimension, i % self.dimension] == 0:
                ap -= 1
            i += 1
        self.board[i // self.dimension, i % self.dimension] = -1

    def _contract(self):
        self.board[self.elbows[0][0], self.elbows[0][1]] = 0

        self.elbows[0] += np.sign(self.elbows[1] - self.elbows[0])
        if np.all(self.elbows[0] == self.elbows[1]):
            self.elbows = self.elbows[1:]