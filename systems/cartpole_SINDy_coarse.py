from __future__ import division, print_function

import numpy as np
import torch
from mars import config, DeterministicFunction

import pickle
from .systems_utils import predict_tensor

# Load the SINDy model ##########################################################################
with open('SINDy_models/model_cartpole_sindy_coarse_2', 'rb') as file:
    model = pickle.load(file)

feature_names = model["feature_names"]
n_features = len(feature_names)
for i in range(n_features):
    feature_names[i] = feature_names[i].replace(" ", "*")
    feature_names[i] = feature_names[i].replace("^", "**")
    feature_names[i] = feature_names[i].replace("sin", "torch.sin")
    feature_names[i] = feature_names[i].replace("cos", "torch.cos")

coefficients = model["coefficients"]

cp_quantile = model["model_error"]['quantile']
print("cp_quantile = ", cp_quantile)
#################################################################################################

class CartPole_SINDy_coarse(DeterministicFunction):
    """
    Parameters
    ----------
    dt : float, optional
        The sampling period used for discretization.
    normalization : tuple, optional
        A tuple (Tx, Tu) of 1-D arrays or lists used to normalize the state and
        action, such that x = diag(Tx) * x_norm and u = diag(Tu) * u_norm.

    """

    def __init__(self, dt=0.01, normalization=None):
        """Initialization; see `CartPole`.""" 
        super(CartPole_SINDy_coarse, self).__init__()
        # TODO: make these inputs of the constructor
        self.feature_names = feature_names
        self.coefficients = coefficients
        self.cp_quantile = cp_quantile

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
        A : ndarray
            The linearized state matrix.
        B : ndarray
            The linearized action matrix.
        """
        #TODO: debug ##################################################################
        # Equilibrium point
        x_eq = torch.zeros((1, self.state_dim))
        u_eq = torch.zeros((1, self.action_dim))

        # Open-loop true dynamics: x_dot = f(x,u)
        f = lambda x, u: self.ode(x, u).squeeze()

        # Evaluate the function at x0
        f_eq = f(x_eq, u_eq)
        
        # Initialize the Jacobian matrix
        A = np.zeros((self.state_dim, self.state_dim))
        B = np.zeros((self.state_dim, self.action_dim))
        
        delta = 1e-5

        for i in range(self.state_dim):
            x_perturbed = x_eq.detach().clone()
            for j in range(self.state_dim):
                x_perturbed[0][j] += delta
                f_perturbed = f(x_perturbed, u_eq)
                A[i, j] = (f_perturbed[i] - f_eq[i]) / delta
                x_perturbed[0][j] = x_eq[0][j]  # Reset the perturbed value

        for i in range(self.state_dim):
            u_perturbed = u_eq.detach().clone()
            for j in range(self.action_dim):
                u_perturbed[0][j] += delta
                f_perturbed = f(x_eq, u_perturbed)
                B[i, j] = (f_perturbed[i] - f_eq[i]) / delta
                u_perturbed[0][j] = u_eq[0][j]  # Reset the perturbed value        

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
        # Compute f(x, u) using the SINDy model
        state_derivative_sindy = predict_tensor(state, action, self.feature_names, self.coefficients)

        #state2 = state.clone()
        #action2 = action.clone()
        #Theta = model.get_regressor(state2.detach().numpy(), action2.detach().numpy())
        #coeff = self.coefficients
        #state_derivative_sindy2 = Theta @ coeff.T
        #state_derivative_sindy2 = torch.tensor(state_derivative_sindy2) # Convert AxesArray to tensor
        
        #assert torch.all(abs(state_derivative_sindy2 - state_derivative_sindy) < 1e-5)

        return state_derivative_sindy
    
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

