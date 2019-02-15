import Deterministic_GridWorld
import Stochastic_Policy
import numpy as np


def main():
    env = Deterministic_GridWorld.GridWorld()
    pol = Stochastic_Policy.Stochastic_Policy() # The policy to be evaluated

    alpha = 0.001
    gamma = 1
    num_training_episodes = 100000

    num_actions = len(env.get_actions())
    num_rows = env.get_num_rows()
    num_cols = env.get_num_cols()

    # Initialize action-value estimate Q(S, A) to zeros
    Q = np.zeros([num_rows, num_cols, num_actions])

    for _ in range(num_training_episodes):
        state = env.reset() # Initialize S_0
        action = pol.select_action(state) # Choose action A according to pi
        done = False
        while not done:
            next_state, reward, done = env.step(action) # Take action A, observe R and S'
            next_action = pol.select_action(state) # Choose A' from S'
            # Update action-value estimates
            Q[state[0]][state[1]][action] = Q[state[0]][state[1]][action] \
                + alpha * (reward + gamma * Q[next_state[0]][next_state[1]][next_action] - Q[state[0]][state[1]][action])
            state = next_state
            action = next_action
    
    print(Q)
           

if __name__ == '__main__':
    main()