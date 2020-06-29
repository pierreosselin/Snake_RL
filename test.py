import unittest
from snake import Snake
import numpy as np

class MyTestCase(unittest.TestCase):
    def test_border(self):
        env = Snake(7, 3, -10, 10)
        env.act(0)
        env.act(0)
        _, _, _, _, done = env.act(0)
        self.assertEqual(done, True)

        env = Snake(8, 2, -10, 10)
        env.act(1)
        env.act(1)
        env.act(1)
        _, _, _, _, done = env.act(1)
        self.assertEqual(done, True)

        env = Snake(7, 5, -10, 10)
        _, _, _, _, done = env.act(2)
        self.assertEqual(done, True)

        env = Snake(5, 2, -10, 10)
        env.act(3)
        env.act(3)
        _, _, _, _, done = env.act(3)
        self.assertEqual(done, True)

    def test_collision(self):
        env = Snake(7, 5, -10, 10)
        env.act(1)
        env.act(2)
        _, _, _, _, done = env.act(3)
        self.assertEqual(done, True)

        size = 5
        while size > 4:
            env = Snake(7, 4, -10, 10)
            env.act(1)
            env.act(2)
            _, _, _, _, done = env.act(3)
            size = env.size
        self.assertEqual(done, False)

    def test_sample_apple(self):
        for i in range(25):
            env = Snake(2, 2, -10, -10)
            self.assertEqual(np.any(env.board == 1), True)

    def test_apple(self):
        env = Snake(2, 2, -10, 10)
        env.act(3)
        _, _, _, rwd, _ = env.act(2)
        self.assertEqual(rwd, 10)

if __name__ == '__main__':
    unittest.main()
