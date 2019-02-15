# A simple policy to be evaluated
# Always select the action that is in the same direction as the velocity
def simple_policy(velocity):
    if velocity > 0:
        return 1
    elif velocity < 0:
        return -1
    else:
        return 0  