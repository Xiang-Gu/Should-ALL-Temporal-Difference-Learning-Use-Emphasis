import numpy as np
from simple_policy import simple_policy


class MountainCar:
    # Three legal actions for this environment
    # 1, -1, 0 represents full throttle forward,
    # full throttle reverse, and zero throttle respectively
    actions = [1, -1, 0] 

    def __init__(self):
        # np.random.seed(0)
        # Each episode starts from a random position in [-0.6, 0.4) and zero velocity
        position = -0.6 + np.random.random() * 0.2 
        self._state = (position, 0.0)

    def step(self, action):
        # Check for validity of action
        assert action in MountainCar.actions

        position, velocity = self._state[0], self._state[1]
        reward = -1
        done = False

        # Update velocity according to a simplied physics
        velocity += 0.001 * action - 0.0025 * np.cos(3 * position)
        # Bound velocity to the range [-0.07, 0.07]
        if velocity < -0.07: velocity = -0.07
        elif velocity > 0.07: velocity = 0.07 
        
        # Update position
        position += velocity
        # Bound position in the range [-1.2, 0.5]
        if position >= 0.5:
            position = 0.5
            done = True
        elif position < -1.2:
            position = -1.2
            velocity = 0.0
        
        self._state = (position, velocity)
        return self._state, reward, done
    
    def reset(self):
        # np.random.seed(0)
        position = -0.6 + np.random.random() * 0.2 
        self._state = (position, 0.0)
        return self._state
    
    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        return self._state


# Test code
# if __name__ == '__main__':
#     env = MountainCar()
#     print('initial state is: ' + str(env.get_state()))

#     print('Move 5 steps')

#     for idx in range(5):
#         print('state before move is: ' + str(env.get_state()))
#         action = simple_policy(env.get_state()[1])
#         print(str(idx) + ' action is: ' + str(action))
#         env.step(action)
#         print('state after move is: ' + str(env.get_state()) + '\n\n')
    
#     print('5 steps done')

    
    


            