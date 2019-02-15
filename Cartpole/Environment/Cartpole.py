from numpy import sin, cos, pi

''' Classic Cartpole Task
    Reference: Bartol, A., Sutton, R. S., & Anderson, C. W. (1983). Neuro-like adaptive elements that can solve difficult learning problems. IEEE Trans. Syst., Man, Cybern, 13, 834-846.'''
num_actions = 2

class Cartpole:
    '''
    states: [x, theta, dx, dtheta]
    actions: [-10., +10] (Forces in Newton. Actual actions might be indexed [0, 1])
    Transition is simulated by a simplified physics
    '''

    dt = .02

    GRAVITY = -9.8 # gravitational acceleration [m * s ** -2]
    CART_MASS = 1. # Mass of cart [kg]
    POLE_MASS = .1 # Mass of pole [kg]
    HALF_POLE_LENGTH = .5 # Half-pole length [m]
    FRICTION_CART = .0005 # Coefficient of friction of cart on track
    FRICTION_POLE = .000002 # Coefficient of friction of pole on cart
    AVAIL_FORCE = [-10., +10.] # Available force applied to the center of the cart [Newton]

    MAX_TRACK_LENGTH = 2.4 # Max length of track; Considered failure if cart moves beyong this range [m]
    MAX_THETA = 12 * (2 * pi / 360) # Max angle of pole; Considered failure if pole falls over this angle [radian]


    def __init__(self):
        # The cart starts at the middle of the track and the pole starts upright
        self._state = [0., 0., 0., 0.]

    def step(self, action):
        x, theta, dx, dtheta = self._state
        force = self.AVAIL_FORCE[action]

        g = self.GRAVITY
        mc = self.CART_MASS
        m = self.POLE_MASS
        l = self.HALF_POLE_LENGTH
        muc = self.FRICTION_CART
        mup = self.FRICTION_POLE

        ddtheta = (g * sin(theta) + cos(theta) * ((-force - m * l * dtheta ** 2 * sin(theta) + muc * sgn(dx)) / (mc + m)) - (mup * dtheta / (m * l))) \
            / (l * (4 / 3 - (m * cos(theta) ** 2) / (mc + m)))

        ddx = (force + m * l * (dtheta ** 2 * sin(theta) - ddtheta * cos(theta)) - muc * sgn(dx)) / (mc + m)

        # Kinematics: Euler integration
        x += dx * self.dt
        theta += dtheta * self.dt
        dx += ddx * self.dt
        dtheta += ddtheta * self.dt

        self._state = [x, theta, dx, dtheta]
        reward = +1.
        done = self._terminal()
        return self._state, reward, done

    def reset(self):
        self.__init__()
        return self._state

    def get_state(self):
        return self._state

    def set_state(self, state):
        self._state = state
        return self._state

    # Determine whether self.state has reached the goal
    def _terminal(self):
        s = self._state
        done = s[0] < -self.MAX_TRACK_LENGTH \
            or s[0] > self.MAX_TRACK_LENGTH \
            or s[1] < -self.MAX_THETA \
            or s[1] > self.MAX_THETA
        return bool(done)

# Extract sign of input scaler x (a real number)
def sgn(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
