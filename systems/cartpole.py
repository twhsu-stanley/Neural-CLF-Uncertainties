from __future__ import division, print_function

import numpy as np
import torch
from mars import config, DeterministicFunction 

class CartPole(DeterministicFunction):
    """
    Parameters
    ----------
    pendulum_mass : float
    cart_mass : float
    length : float
    dt : float, optional
        The sampling period used for discretization.
    normalization : tuple, optional
        A tuple (Tx, Tu) of 1-D arrays or lists used to normalize the state and
        action, such that x = diag(Tx) * x_norm and u = diag(Tu) * u_norm.

    """

    def __init__(self, pendulum_mass, cart_mass, length, friction=0.0, 
                dt=0.01, normalization=None):
        """Initialization; see `CartPole`.""" 
        super(CartPole, self).__init__()
        self.pendulum_mass = pendulum_mass
        self.cart_mass = cart_mass
        self.length = length
        self.friction = friction
        self.dt = dt
        self.gravity = 9.81
        self.state_dim = 4
        self.action_dim = 1
        self.normalization = normalization
        if normalization is not None:
            self.normalization = [np.array(norm, dtype=config.np_dtype)
                                  for norm in normalization]
            self.inv_norm = [norm ** -1 for norm in self.normalization]

    def normalize(self, state, action):
        """Normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
        state = torch.mm(state, torch.tensor(Tx_inv, device = config.device))

        if action is not None:
            action = torch.mm(action, torch.tensor(Tu_inv, device = config.device))
        
        return state, action

    def denormalize(self, state, action):
        """De-normalize states and actions."""
        if self.normalization is None:
            return state, action

        Tx, Tu = map(np.diag, self.normalization)
        state = torch.mm(state, torch.tensor(Tx, device = config.device))
        if action is not None:
            action = torch.mm(action, torch.tensor(Tu, device = config.device))

        return state, action

    def linearize_ct(self):
        """Return the discretized, scaled, linearized system.

        Returns
        -------
        Ad : ndarray
            The discrete-time state matrix.
        Bd : ndarray
            The discrete-time action matrix.

        """
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.friction
        g = self.gravity

        A = np.array([[0, 0,                   1, 0                            ],
                    [0, 0,                     0, 1                            ],
                    [0, m*g/M,              -b/M, 0                            ],
                    [0, g * (m + M) / (L * M), -b/(M*L), 0                     ]],
                    dtype=config.np_dtype)

        B = np.array([0, 0, 1/M, 1 / (M * L)]).reshape((-1, self.action_dim))

        if self.normalization is not None:
            Tx, Tu = map(np.diag, self.normalization)
            Tx_inv, Tu_inv = map(np.diag, self.inv_norm)
            A = np.linalg.multi_dot((Tx_inv, A, Tx))
            B = np.linalg.multi_dot((Tx_inv, B, Tu))

        return A, B

    def ode(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        state: ndarray or Tensor
            States.
        action: ndarray or Tensor
            Actions.

        Returns
        -------
        state_derivative: Tensor
            The state derivative according to the dynamics.

        """
        # Physical dynamics
        m = self.pendulum_mass
        M = self.cart_mass
        L = self.length
        b = self.friction
        g = self.gravity

        x, theta, v, omega = torch.split(state, [1, 1, 1, 1], dim=1)

        x_dot = v
        theta_dot = omega

        det = M + m * torch.mul(torch.sin(theta), torch.sin(theta))
        v_dot = (action - b * v - m * L * torch.mul(omega, omega) * torch.sin(theta)  + 0.5 * m * g * torch.sin(2 * theta)) / det
        omega_dot = (action * torch.cos(theta) - 0.5 * m * L * torch.mul(omega, omega) * torch.sin(2 * theta) - b * torch.mul(v, torch.cos(theta))
                    + (m + M) * g * torch.sin(theta)) / (det * L)

        state_derivative = torch.cat((x_dot, theta_dot, v_dot, omega_dot), dim=1)

        return state_derivative
    
    def ode_normalized(self, state, action):
        """Compute the state time-derivative.

        Parameters
        ----------
        states: ndarray or Tensor
            Unnormalized states.
        actions: ndarray or Tensor
            Unnormalized actions.

        Returns
        -------
        x_dot: Tensor
            The normalized derivative of the dynamics

        """
        
        state, action = self.denormalize(state, action)
        state_derivative = self.ode(state, action)

        # Normalize
        return self.normalize(state_derivative, None)[0]
