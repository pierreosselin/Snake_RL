"""
File to implement the snake environment
"""


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

    State : numpy array RxC, -1 = Snake, 0 = Nothing, 1 = Apple
    """

    def __init__(self, n_columns, n_rows, init_size):
        self.n_columns = n_columns
        self.n_rows = n_rows
        self.init_size = init_size



