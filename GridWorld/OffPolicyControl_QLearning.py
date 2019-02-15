import Deterministic_GridWorld
import Stochastic_Policy
import numpy as np


def main():
    env = Deterministic_GridWorld.GridWorld()
    pol = Stochastic_Policy.Stochastic_Policy() # The policy to be evaluated

    alpha = 0.001
    gamma = 1
    epsilon = 0.1
    num_training_episodes = 100000

    num_actions = len(env.get_actions())
    num_rows = env.get_num_rows()
    num_cols = env.get_num_cols()

    # Initialize action-value estimate Q(S, A) to zeros
    Q = np.zeros([num_rows, num_cols, num_actions])

    for _ in range(num_training_episodes):
        state = env.reset() # Initialize S_0
        done = False
        while not done:
            action = pol.epsilon_greedy(Q[state[0]][state[1]], epsilon) # Choose A from S epsilon-greedily from Q(S, A)
            next_state, reward, done = env.step(action) # Take action A, observe R and S'
            # Find the greedy action in next_state
            action_values_next_state = Q[next_state[0]][next_state[1]]
            greedy_action_next_state = action_values_next_state.tolist().index(max(action_values_next_state))
            # Update action-value estimate (Q) for current state-action pair on the basis of
            # the max Q-values for the next state S', assuming the agent were to follow a greedy policy from S'
            Q[state[0]][state[1]][action] = Q[state[0]][state[1]][action] \
                + alpha * (reward + gamma * Q[next_state[0]][next_state[1]][greedy_action_next_state] - Q[state[0]][state[1]][action])
            state = next_state
    
    # Extract the policy from the Q-values to see if we've found the correct optimal policy
    optimal_actions = Stochastic_Policy.extract_optimal_actions(Q)
    print(optimal_actions)

    
if __name__ == '__main__':
    main()