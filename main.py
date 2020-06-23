"""
Main File
"""
from snake import Snake
from agent import DQN
from tqdm import tqdm

def evaluate_agent(env, agent):
    state, done = env.reset(evaluate=True, name_video="Output1")
    total_rwd = 0
    # Sample a trajectory
    while not done:
        action = agent.learned_act(state)
        state, action, next_state, rwd, done = env.act(action)
        state = next_state
        total_rwd += rwd

    print("Total Reward of the evaluation:", rwd)
def main():

    EPOCHS = 500
    DIMENSION = 5
    REWARD_LOSS = -1
    REWARD_GAIN = 1
    REWARD_NOTHING = -1/(DIMENSION**2)
    INIT_SIZE = 3
    BUFFER_SIZE = 1000
    BATCH_SIZE = 64
    EPSILON = 0.1
    MAX_ITERATION = 200

    agent = DQN(DIMENSION, BUFFER_SIZE, BATCH_SIZE, EPSILON)

    for _ in tqdm(range(EPOCHS)):
        """
        1) Instantiate environment
        2) Play a whole game with agent
        3) Perform one model update
        """
        # Instantiate env
        env = Snake(DIMENSION, INIT_SIZE, REWARD_LOSS, REWARD_GAIN, REWARD_NOTHING, MAX_ITERATION)
        state, done = env.reset()
        total_rwd = 0.
        # Sample a trajectory
        while not done:
            action = agent.predict(state)
            state, action, next_state, rwd, done = env.act(action)
            agent.update_buffer([state, action, next_state, rwd, done])
            state = next_state
            total_rwd += rwd

        print("Total Reward of the trajectory:", total_rwd)
        # Perform model update
        agent.fit()

    print("Evaluation of the Agent...")
    evaluate_agent(env, agent)

if __name__ == '__main__':
    main()