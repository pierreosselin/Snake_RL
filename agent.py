import numpy as np
import random
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import RMSprop, SGD
from collections import deque

class Agent:
    """
    Implement the Agent class

    methods:
    -

    attributes:
    - model

    """

    def __init__(self):
        pass

    def create_model(self):
        pass

    def fit(self):
        pass


class DQN(Agent):

    def __init__(self, input_dimension, buffer_size, batch_size, epsilon, n_action=4, gamma=0.99):
        super().__init__()
        self.input_dimension = input_dimension + 2
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_action = n_action
        self.buffer = []
        self.num_last_frames = 4
        self.model = self._create_model(self.num_last_frames)
        self.model.summary()
        self.t = 0
        self.last_frames = None

    def _create_model(self, num_last_frames):
        pass

    def reset_frames(self):
        self.last_frames = None

    def transform_board(self, board):
        """
        Transform the current board into the current state, depends on the model implemented
        """
        pass

    def _update_frames(self, board):
        element = board
        if self.last_frames is None:
            self.last_frames = deque([element] * self.num_last_frames)
        else:
            self.last_frames.pop()
            self.last_frames.appendleft(element)

    def _sample_batch(self):
        if self.batch_size <= len(self.buffer):
            return random.sample(self.buffer, self.batch_size)
        else:
            return self.buffer

    def update_buffer(self, element):
        if len(self.buffer) == self.buffer_size:
            del self.buffer[0]
            self.buffer.append(element)
        else:
            self.buffer.append(element)

    def predict(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(4)
        else:
            return np.argmax(self.model.predict(state[None]))

    def learned_act(self, state):
        return np.argmax(self.model.predict(state[None]))

    def fit(self):
        batch = self._sample_batch()
        self.epsilon = (10 / (self.t + 100)) + 0.05
        self.t += 1

        target = np.zeros((self.batch_size, self.n_action))
        state_input = np.zeros((self.batch_size, self.num_last_frames, self.input_dimension, self.input_dimension))
        
        for i, [state, action, next_state, reward, done] in enumerate(batch):
            state_input[i] = state
            target[i] = self.model.predict(state[None])[0]
            
            if not done:
                target[i, action] = reward + self.gamma * np.max(self.model.predict(next_state[None]))
            else:
                target[i, action] = reward
        self.model.train_on_batch(x=state_input, y=target)

class DQN_CNN(DQN):

    def __init__(self, input_dimension, buffer_size, batch_size, epsilon, n_action=4, gamma=0.99):
        super(DQN_CNN, self).__init__(input_dimension, buffer_size, batch_size, epsilon, n_action, gamma)

    def _create_model(self, num_last_frames):
        model = Sequential([
            Conv2D(16, kernel_size=(3, 3), strides=(1, 1), data_format='channels_first', input_shape=(num_last_frames, self.input_dimension, self.input_dimension), activation='relu'),
            Conv2D(32, kernel_size=(3, 3), strides=(1, 1), data_format='channels_first', activation='relu'),
            Flatten(),
            Dense(256, activation="relu"),
            Dense(self.n_action)
        ])

        model.compile(RMSprop(), "MSE")

        return model

    def transform_board(self, board):
        """
        Transform the current board into the current state
        """
        self._update_frames(board)
        return np.array(self.last_frames)


class DQN_FNN(DQN):

    def __init__(self, input_dimension, buffer_size, batch_size, epsilon, n_action=4, gamma=0.99):
        super(DQN_FNN, self).__init__(input_dimension, buffer_size, batch_size, epsilon, n_action, gamma)

    def _create_model(self, num_last_frames):
        model = Sequential([
            Conv2D(64, 3, input_shape=(self.input_dimension, self.input_dimension, 3), activation='relu'),
            Flatten(),
            Dense(100, activation="relu"),
            Dense(self.n_action)
        ])

        model.compile(SGD(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        return model

class DQN_FC(DQN):
    def __init__(self, input_dimension, buffer_size, batch_size, epsilon, n_action=4, gamma=0.99):
        super(DQN_FC, self).__init__(input_dimension, buffer_size, batch_size, epsilon, n_action, gamma)

    def _create_model(self, num_last_frames):
        model = Sequential([
            Flatten(input_shape=(self.input_dimension, self.input_dimension, 3)),
            Dense(100, activation="relu"),
            Dense(100, activation="relu"),
            Dense(30, activation="relu"),
            Dense(self.n_action)
        ])

        model.compile(SGD(lr=0.1, decay=1e-4, momentum=0.0), "mse")
        return model