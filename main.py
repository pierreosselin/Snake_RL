"""
Main File
"""
from snake import Snake
from agent import DQN_CNN, DQN_FNN, DQN_FC
from tqdm import tqdm
import numpy as np
import keras

def evaluate_agent(env, agent, name):
    board, done = env.reset(evaluate=True, name_video=name)
    state = agent.transform_board(board)
    total_rwd = 0
    # Sample a trajectory
    while not done:
        action = agent.learned_act(state)
        board, action, next_board, rwd, done = env.act(action)
        next_state = agent.transform_board(next_board)
        state = next_state
        total_rwd += rwd

    print("Total Reward of the evaluation:", total_rwd)
def main():

    EPOCHS = 2000
    DIMENSION = 5
    REWARD_LOSS = -1
    REWARD_GAIN = 2
    REWARD_NOTHING = -1/(DIMENSION**2)
    INIT_SIZE = 3
    BUFFER_SIZE = 1000
    BATCH_SIZE = 32
    EPSILON = 0.1
    MAX_ITERATION = 200
    AGENT = "DQN_CNN"

    agent_dict = {"DQN_CNN": DQN_CNN, "DQN_FC": DQN_FC, "DQN_FNN": DQN_FNN}

    agent = agent_dict[AGENT](DIMENSION, BUFFER_SIZE, BATCH_SIZE, EPSILON)

    for _ in tqdm(range(EPOCHS)):
        """
        1) Instantiate environment
        2) Play a whole game with agent
        3) Perform one model update
        """
        # Instantiate env
        env = Snake(DIMENSION, INIT_SIZE, REWARD_LOSS, REWARD_GAIN, REWARD_NOTHING, MAX_ITERATION)
        board, done = env.reset()
        state = agent.transform_board(board)
        total_rwd = 0.
        # Sample a trajectory
        while not done:
            action = agent.predict(state)
            board, action, next_board, rwd, done = env.act(action)
            next_state = agent.transform_board(next_board)
            agent.update_buffer([state, action, next_state, rwd, done])
            if rwd > 0:
                agent.update_buffer([np.rot90(state, k=1, axes=(1, 2)), (action - 1) % 4, np.rot90(next_state, k=1, axes=(1, 2)), rwd, done])
                agent.update_buffer([np.rot90(state, k=2, axes=(1, 2)), (action - 2) % 4, np.rot90(next_state, k=2, axes=(1, 2)), rwd, done])
                agent.update_buffer([np.rot90(state, k=3, axes=(1, 2)), (action - 3) % 4, np.rot90(next_state, k=3, axes=(1, 2)), rwd, done])

            state = next_state
            total_rwd += rwd

        print("Total Reward of the trajectory:", total_rwd)
        # Perform model update
        agent.fit()
        agent.reset_frames()
    print("Saving Model...")
    agent.model.save(AGENT)

    print("Evaluation of the Agent...")
    evaluate_agent(env, agent, "Output" + AGENT)

def eval():

    DIMENSION = 5
    REWARD_LOSS = -1
    REWARD_GAIN = 2
    REWARD_NOTHING = -1/(DIMENSION**2)
    INIT_SIZE = 3
    MAX_ITERATION = 200

    env = Snake(DIMENSION, INIT_SIZE, REWARD_LOSS, REWARD_GAIN, REWARD_NOTHING, MAX_ITERATION)

    model = keras.models.load_model("modelFC")
    state, done = env.reset(evaluate=True, name_video="OutputFC")

    total_rwd = 0
    # Sample a trajectory
    while not done:
        action = np.argmax(model.predict(state[None]))
        state, action, next_state, rwd, done = env.act(action)
        state = next_state
        total_rwd += rwd

    print("Total Reward of the evaluation:", total_rwd)

if __name__ == '__main__':
    main()