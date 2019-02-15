import numpy as np
from time import sleep

class PuddleWorld:
    # Four legal actions for this environment
    # 0, 1, 2, 3 represents up, down, left, right respectively
    actions = [0, 1, 2, 3]
    # Two puddles that are 0.1 wide from its centers from (0.1,0.75) to (0.45,0.75)
    # and (0.45, 0.4, 0.45, 0.8)
    # [x_1, y_1, x_2, y_2, width]
    puddles = [[0.1, 0.75, 0.45, 0.75, 0.1], [0.45, 0.4, 0.45, 0.8, 0.1]]

    def __init__(self):
        # np.random.seed(0)
        # Each episode starts from a random position in non-goal region
        # while True:
        #     x = np.random.random()
        #     y = np.random.random()
        #     if y < 1.9 - x: # y >= 1.9 - x (0 <= x <= 1.0, 0 <= y <= 1.0) is the goal region
        #         break
        # self._state = (x, y)

        # Initialize state to be the vicinity of (0.33, 0.5)
        x = 0.33
        y = 0.5
        dx = np.random.normal(0, 0.01)
        dy = np.random.normal(0, 0.01)
        x = min(1.0, max(0.0, x + dx))
        y = min(1.0, max(0.0, y + dy))
        self._state = (x, y)

    def step(self, action):
        # Check for validity of action
        assert action in PuddleWorld.actions

        x = self._state[0]
        y = self._state[1]
        done = False
        dx = np.random.normal(0, 0.01) # Each move is associated with a random noise in both motions
        dy = np.random.normal(0, 0.01)


        if action == 0: dy += 0.05 # Up
        elif action == 1: dy -= 0.05 # Down
        elif action == 2: dx -= 0.05 # Left
        else: dx += 0.05 # Right
        x = min(1.0, max(0.0, x + dx)) # Bound x and y within [0.0, 1.0]
        y = min(1.0, max(0.0, y + dy))

        # Check if it enters goal region after the move
        done = y >= 1.9 - x

        reward = -1.0
        for puddle in PuddleWorld.puddles:
            x_1, y_1, x_2, y_2, width = puddle
            reward += 400 * min(0.0, self.distance_lineseg_point(x, y, x_1, y_1, x_2, y_2, width))

        self._state = (x, y)
        return self._state, reward, done

    def reset(self):
        self.__init__()
        return self._state

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        return self._state

    # Compute how far you are from (x,y) to the puddle
    # If it returns a negative value (say -0.05) then it means (x,y) are within the puddle and the distance to
    # the nearest edge is 0.05
    # If it returns a position value (say 0.1) then it means (x,y) is out of the puddle and it is 0.1 far away
    def distance_lineseg_point(self, x, y, x_1, y_1, x_2, y_2, width):
        # Horizontal puddle
        if y_1 == y_2:
            if x <= x_1:
                return np.sqrt((x-x_1)**2 + (y-y_1)**2) - width
            elif x >= x_2:
                return np.sqrt((x-x_2)**2 + (y-y_2)**2) - width
            else:
                return abs(y - y_1) - width
        # Verticial puddle
        elif x_1 == x_2:
            if y <= y_1:
                return np.sqrt((x-x_1)**2 + (y-y_1)**2) - width
            elif y >= y_2:
                return np.sqrt((x-x_2)**2 + (y-y_2)**2) - width
            else:
                return abs(x - x_1) - width
        else:
            assert False, 'Unrecognized puddle shape (only horizontal and vertical is accepted)'

# Test code
if __name__ == '__main__':
    print('initial state is: ' + str(env.get_state()))

    print('Move 5 steps')

    for idx in range(5):
        print('state before move is: ' + str(env.get_state()))
        action = np.random.choice([0,1,2,3])
        print(str(idx) + ' action is: ' + str(action))
        env.step(action)
        print('state after move is: ' + str(env.get_state()) + '\n\n')

    print('5 steps done')
