import numpy as np
from numpy import sin, cos, pi
# from scipy.integrate import solve_ivp

''' Classic Acrobot Task'''
num_actions = 3

class Acrobot:
    '''
    states: [theta1, theta2, dtheta1, dtheta2]
    actions: [-1. 0., +1.]
    Transition is simulated by a simplified physics
    '''

    dt = .2

    LINK_MASS_1 = 1. # Mass of link 1 [kg]
    LINK_MASS_2 = 1. # Mass of link 2 [kg]
    LINK_LENGTH_1 = 1. # Length of link 1 [m]
    LINK_LENGTH_2 = 1. # Length of link 2 [m]
    LINK_LENGTH_COM_1 = .5 # Length to center of mass of link 1 [m]
    LINK_LENGTH_COM_2 = .5 # Length to center of mass of link 2 [m]
    LINK_MOI_1 = 1. # Moment of inertia of link 1 [kg * m^2]
    LINK_MOI_2 = 1. # Moment of inertia of link 2 [kg * m^2]
    GRAVITY = 9.8 # Gravity [m * s^-2]

    # If the following two quantities are change,
    # CHANGE TILECODER AS WELL (SCALERFACTOR!)
    MAX_VEL_1 = 4 * pi # Maximal angular velocity of link 1 (minimal value is -MAX_VEL_1)
    MAX_VEL_2 = 9 * pi # Maximal angular velocity of link 2 (minimal value is -MAX_VEL_2)

    AVAIL_TORQUE = [-1., 0., +1.] # Three legal actions for this problem

    def __init__(self):
        # The acrobot starts with both links hanging downwards without any velocity
        self._state = [0., 0., 0., 0.]

    def step(self, action):
        s = self._state
        torque = self.AVAIL_TORQUE[action]
        s_augmented = np.append(s, torque)

        # # UTCS Lab machine does not support scipy
        # sol = solve_ivp(self._dydt, [0., self.dt], s_augmented) # solve_ivp(fun, t_span, initial_state)
        # # We only care about the state in the last time step
        # next_state = sol.y[:, -1]
        # next_state = next_state[:4] # Omit action

        next_state = rk4(self._dydt, 0.0, s_augmented, self.dt) # Here we always use t0 = 0.0 which does not seem reasonable at first sight. However, it works because the derivs is indepedent with t. That is, Acrobot is a time-invariant system.
        next_state = next_state[:-1] # Omit action

        # Enforce bound check
        next_state[0] = wrap(next_state[0], 0, 2 * pi)
        next_state[1] = wrap(next_state[1], 0, 2 * pi)
        next_state[2] = bound(next_state[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        next_state[3] = bound(next_state[3], -self.MAX_VEL_2, self.MAX_VEL_2)

        self._state = next_state
        done = self._terminal()
        reward = -1. if not done else 0.
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
        return -cos(s[0]) - cos(s[0] + s[1]) >= 1.0 # Justify this yourself using Geometry!

    # compute the derivative of y w.r.t t
    def _dydt(self, t, s_augmented):
        m1 = self.LINK_MASS_1
        m2 = self.LINK_MASS_2
        l1 = self.LINK_LENGTH_1
        l2 = self.LINK_LENGTH_2
        lc1 = self.LINK_LENGTH_COM_1
        lc2 = self.LINK_LENGTH_COM_2
        I1 = self.LINK_MOI_1
        I2 = self.LINK_MOI_2
        g = self.GRAVITY

        theta1 = s_augmented[0]
        theta2 = s_augmented[1]
        dtheta1 = s_augmented[2]
        dtheta2 = s_augmented[3]
        torque = s_augmented[4]

        phi2 = m2 * lc2 * g * cos(theta1 + theta2 - 0.5 * pi)
        phi1 = -m2 * l1 * lc2 * dtheta2 ** 2 * sin(theta2) \
            - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sin(theta2) \
            + (m1 * lc1 + m2 * l1) * g * cos(theta1 - 0.5 * pi) + phi2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * cos(theta2)) + I2
        d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * cos(theta2)) \
            + I1 + I2
        ddtheta2 = (torque + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * sin(theta2) - phi2) \
            / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1) # Use the equation in the book rather than the NIPS paper!
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1

        return [dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0]


# Bound input x to the range [m, M]
def bound(x, m, M):
    assert m<=M, 'lower bound should <= higher bound'
    return max(m, min(M, x))

# Wrap x in the range [m, M]
def wrap(x, m, M):
    assert m <= M, 'lower bound should <= higher bound'
    diff = M - m
    if x > M:
        return m + (x - M) % diff
    elif x < m:
        return M - (m - x) % diff
    else:
        return x

# Compute the numerical integration of y(t = t0+dt) given y(0) and dy/dt
# Note y0 is a vector of initial values
def rk4(derivs, t0, y0, dt):
    '''
    derivs is a function that returns the derivatives of y w.r.t t (i.e. right-hand side of the ODEs)
    it has the signature derivs(t, y).
    If y0 is a vector [y_1(t0), y_2(t0), ..., y_n(t0)], then derivs should also return a vector of length n where
    the ith component is the derivative of y_i(t) w.r.t. t. And this function will return a vector of length n
    where the ith component is the approximated value of y_i(t0 + dt)
    '''
    k1 = np.asarray(derivs(t0, y0))
    k2 = np.asarray(derivs(t0 + dt / 2, y0 + k1 * dt / 2)) # This gives you an approximated derivative of y w.r.t t at time t0+dt/2, because y0 + k1 * dt/2 is an estimate of the real y(t0 + dt/2)
    k3 = np.asarray(derivs(t0 + dt / 2, y0 + k2 * dt / 2)) # This gives another estimate of derivative of y w.r.t t at time t0+dt/2, because now the real y(t0 + dt/2) is approximated by y0 + k2 * dt/2
    k4 = np.asarray(derivs(t0 + dt, y0 + k3 * dt)) # This gives you an estimate of derivative of y w.r.t. t at time t0+dt, because y0 + k3 * dt is an estimate of the raal y(t0+dt)
    return y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4) # Average over all those slopes to compute a (weighted) mean slope and approximate the next state using this mean slope
