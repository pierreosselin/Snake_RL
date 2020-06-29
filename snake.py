"""
File to implement the snake environment
"""
import numpy as np
import cv2
from collections import deque

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

    State : Board 4xDxD for the 4 previous frames
    - EMPTY = 0
    - FRUIT = 1
    - SNAKE_HEAD = 2
    - SNAKE_BODY = 3
    - WALL = 4
    """

    def __init__(self, dimension, init_size, reward_loss, reward_apple, reward_nothing=-0.05, max_iter=100, evaluate=False, name_video=""):
        self.dimension = dimension + 2
        self.init_size = init_size
        self.reward_loss = reward_loss
        self.reward_apple = reward_apple
        self.reward_nothing = reward_nothing
        self.max_iter = max_iter
        self.action_to_np = {0: np.array([-1, 0]), 1: np.array([0, 1]), 2: np.array([1, 0]), 3: np.array([0, -1])}

        error_msg = "The initial size should be smaller than the number of rows"
        assert self.init_size <= self.dimension - 2, error_msg

        self.reset(evaluate=evaluate, name_video=name_video)

    def reset(self, evaluate=False, name_video=""):
        self.evaluate = evaluate
        self.name_video = name_video
        self.size = self.init_size

        # Initial the snake vertically in the middle
        self.board = np.zeros((self.dimension, self.dimension))
        self.board[int(self.dimension / 2) - int(self.init_size / 2), int(self.dimension / 2)] = 2
        self.board[
        (int(self.dimension / 2) - int(self.init_size / 2) + 1):(
                    int(self.dimension / 2) + self.init_size - int(self.init_size / 2)),
        int(self.dimension / 2)] = 3

        # Initial Walls
        self.board[0, :] = 4
        self.board[-1, :] = 4
        self.board[:, 0] = 4
        self.board[:, -1] = 4

        # List of coordinates where the snake has turned
        self.elbows = [
            np.array([int(self.dimension / 2) + self.size - int(self.size / 2) - 1, int(self.dimension / 2)]),
            np.array([int(self.dimension / 2) - int(self.size / 2), int(self.dimension / 2)])]

        self._sample_apple()
        if self.evaluate:
            self.frame = np.zeros((1, 500, 500, 3), dtype=float)
            self.get_frame()

        return self.board, False

    def act(self, action):
        """
        Return state, action, next_state, reward, done
        state : Current state
        action : Action taken
        next_state : Arriving state
        reward : Reward from action
        done : Boolean if game finished
        """
        ## 0 : Top, 1 : Right, 2 : Bottom, 3 : Left
        state = np.copy(self.board)

        self.board[self.elbows[-1][0], self.elbows[-1][1]] = 3

        if np.all(self.action_to_np[action] == np.sign(self.elbows[-1] - self.elbows[-2])):
            self.elbows[-1] += self.action_to_np[action]
        else:
            self.elbows.append(self.elbows[-1] + self.action_to_np[action])

        ## Four possibilities : Either wall, nothing apple or snake
        if self.board[self.elbows[-1][0], self.elbows[-1][1]] == 4:
            next_state, rwd, done = self.board, self.reward_loss, True

        elif self.board[self.elbows[-1][0], self.elbows[-1][1]] == 1:
            self.board[self.elbows[-1][0], self.elbows[-1][1]] = 2
            self.size += 1
            if self.size < (self.dimension - 2) ** 2:
                self._sample_apple()
                next_state, rwd, done = self.board, self.reward_apple, False
            else:
                next_state, rwd, done = self.board, self.reward_apple, True
        else:
            self._contract()
            if self.board[self.elbows[-1][0], self.elbows[-1][1]] == 0:
                self.board[self.elbows[-1][0], self.elbows[-1][1]] = 2
                next_state, rwd, done = self.board, self.reward_nothing, False

            else:
                next_state, rwd, done = self.board, self.reward_loss, True

        if self.evaluate:
            self.get_frame()
            if done or self.frame.shape[0] > self.max_iter:
                self.create_video(self.name_video)
                done = True

        return state, action, next_state, rwd, done

    def _sample_apple(self):
        real_dim = self.dimension - 2
        nb_possibility = real_dim ** 2 - self.size
        ap = np.random.randint(nb_possibility)
        i = 0
        while ap > 0 or self.board[(i // real_dim) + 1, (i % real_dim) + 1] >= 1:
            if self.board[(i // real_dim) + 1, (i % real_dim) + 1] == 0:
                ap -= 1
            i += 1
        self.board[(i // real_dim) + 1, (i % real_dim) + 1] = 1

    def _contract(self):
        self.board[self.elbows[0][0], self.elbows[0][1]] = 0

        self.elbows[0] += np.sign(self.elbows[1] - self.elbows[0])
        if np.all(self.elbows[0] == self.elbows[1]):
            self.elbows = self.elbows[1:]

    def get_frame(self):
        """
        Transform board into a frame to display
        """
        frame = 255 * np.ones((self.dimension - 2, self.dimension - 2, 3))

        for i, row in enumerate(self.board[1:-1, 1:-1]):
            for j, element in enumerate(row):
                if element == 1:
                    frame[i, j, :] = np.array([255, 0, 0])
                elif element == 3:
                    frame[i, j, :] = np.array([0, 255, 0])
                elif element == 2:
                    frame[i, j, :] = np.array([0, 0, 0])
        frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_NEAREST)
        self.frame = np.vstack([self.frame, frame[None]])

    def create_video(self, name):
        out = cv2.VideoWriter(name + '.avi', cv2.VideoWriter_fourcc(*'DIVX'), 1, (500, 500))
        for frame in self.frame:
            frame = frame.astype(np.uint8)
            out.write(frame)
        out.release()
