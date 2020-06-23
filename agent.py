import numpy as np
import random
from keras import Sequential
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.optimizers import SGD

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

    def __init__(self, input_dimension, buffer_size, batch_size, epsilon, n_action = 4, gamma = 0.99):
        super().__init__()
        self.input_dimension = input_dimension
        self.epsilon = epsilon
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.n_action = n_action
        self.buffer = []
        self.model = self._create_model()
        print(self.model.summary())
        self.t = 0

        self.walls = np.zeros((input_dimension, input_dimension))
        self.walls[0, :] = 1
        self.walls[-1, :] = 1
        self.walls[:, 0] = 1
        self.walls[-1, :] = 1
        self.walls = self.walls[:, :, None]

    def _create_model(self):
        model = Sequential([
            Conv2D(32, 3, input_shape=(self.input_dimension, self.input_dimension, 2), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, 3, activation='relu', padding="same"),
            MaxPooling2D(pool_size=(2, 2), padding="same"),
            Conv2D(32, 3, activation='relu', padding="same"),
            MaxPooling2D(pool_size=(2, 2), padding="same"),
            Flatten(),
            Dense(self.n_action)
        ])

        model.compile(SGD(), "mse")
        return model

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
            state = np.append(state[:, :, None], self.walls, axis=2)
            return np.argmax(self.model.predict(state[None]))

    def learned_act(self, state):
        state = np.append(state[:, :, None], self.walls, axis=2)
        return np.argmax(self.model.predict(state[None]))

    def fit(self):
        batch = self._sample_batch()
        self.epsilon = 100/(self.t + 100) + 0.5
        self.t += 1

        target = np.zeros((self.batch_size, self.n_action))
        state_input = np.zeros((self.batch_size, self.input_dimension, self.input_dimension, 2))

        for i, [state, action, next_state, reward, done] in enumerate(batch):
            state_input[i, :, :, 0] = state
            state_input[i, :, :, 1] = self.walls[:, :, 0]
            if not done:
                next_state = np.append(next_state[:, :, None], self.walls, axis=2)
                target[i, action] = reward + self.gamma * np.max(self.model.predict(next_state[None]))
            else:
                target[i, action] = reward

        self.model.fit(x=state_input, y=target, verbose=0)
