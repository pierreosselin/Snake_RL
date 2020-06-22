import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

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



class DQNAgent(Agent):

    def __init__(self, input_dimension):
        super().__init__()
        self.input_dimension = input_dimension


    def create_model():
        model = Sequential()
        model.add(Conv2D(8, 3, input_shape=(self.input_dimension, self.input_dimension, 1)),
        MaxPooling2D(pool_size=pool_size),
        Flatten(),
        Dense(10, activation='softmax'),)