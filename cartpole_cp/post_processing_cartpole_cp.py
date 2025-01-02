import os
import platform
if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    print("KMP_DUPLICATE_LIB_OK set to true in os.environ to avoid warning on Mac")
import numpy as np
import sys
import math
import matplotlib
import collections
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import copy
from time import perf_counter

import torch
import torch.optim as optim

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import mars
from mars import config
from mars.visualization import plot_nested_roas_3D, plot_nested_roas_3D_diagnostic
from mars.roa_tools import *
from mars.dynamics_tools import *
from mars.utils import get_batch_grad, save_lyapunov_nn, load_lyapunov_nn,\
     save_dynamics_nn, load_dynamics_nn, save_controller_nn, load_controller_nn, count_parameters
from mars.controller_tools import pretrain_controller_nn, train_controller_SGD
from mars.parser_tools import getArgs

from examples.systems_config import all_systems 
from examples.example_utils import build_system, VanDerPol, InvertedPendulum, LyapunovNetwork, compute_roa_ct, balanced_class_weights, generate_trajectories, save_dict, load_dict

from systems import CartPole

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import pickle
import warnings
import random
import argparse
warnings.filterwarnings("ignore")

exp_num = 600

# results_dir = '{}/results/exp_{:03d}_keep_eg3'.format(str(Path(__file__).parent.parent), exp_num)
# results_dir = '{}/results/exp_{:02d}_keep_eg3'.format(str(Path(__file__).parent.parent), exp_num)
results_dir = '{}/results/exp_{:02d}'.format(str(Path(__file__).parent.parent), exp_num)

with open(os.path.join(results_dir, "00hyper_parameters.txt"), "r") as f:
    lines = f.readlines()

input_args = []
for line in lines:
    a, b = line.split(" ")
    b = b[0:-2]
    input_args.append(a)
    input_args.append(b)
args = getArgs(input_args)

device = config.device
print('Pytorch using device:', device)

# Set random seed
seed_num = 0
torch.manual_seed(seed_num)
np.random.seed(seed_num)

# Sampling time
dt = args.dt

# Construct the true and nominal systems #############################################################
# State and control limits
x_max = np.float64(1)
theta_max = np.float64(np.pi/6)
v_max = np.float64(1.5)
omega_max = np.float64(1)
u_max = np.float64(10)

state_norm = (x_max, theta_max, v_max, omega_max) # Match the order of the states !!!
action_norm = (u_max,)

#1. Construct the true system
# True system params
m_true = 0.3 # pendulum mass
M_true = 1 # cart mass
l_true = 1 # length
b_true = 0 # friction coeff

# Initialize the true system
true_system = CartPole(m_true, M_true, l_true, b_true, dt, [state_norm, action_norm])

# Open-loop true dynamics
true_dynamics = lambda x, y: true_system.ode_normalized(x, y)

state_dim = true_system.state_dim
action_dim = true_system.action_dim

#2. Construct the nominal system
# Nominal system params
m_nominal = 0.3 # pendulum mass
M_nominal = 1 # cart mass
l_nominal = 1 # length
b_nominal = 0 # friction coeff

# Initialize the nominal system
nominal_system = CartPole(m_nominal, M_nominal, l_nominal, b_nominal, dt, [state_norm, action_norm])

# Open-loop nominal dynamics
nominal_dynamics = lambda x, y: nominal_system.ode_normalized(x, y)

# Set up computation domain and the initial safe set
# State grid
grid_limits = np.array([[-1., 1.], ] * state_dim)
resolution = args.grid_resolution # Number of states divisions each dimension
grid = mars.GridWorld(grid_limits, resolution)
tau = np.sum(grid.unit_maxes) / 2
u_max = true_system.normalization[1].item()
Tx, Tu = map(np.diag, true_system.normalization)
Tx_inv, Tu_inv = map(np.diag, true_system.inv_norm)

# Set initial safe set as a ball around the origin (in normalized coordinates)
cutoff_radius    = 0.4
initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius

# Control Policies ####################################################################################
#1. LQR policy
A, B = nominal_system.linearize_ct()
Q = np.identity(state_dim).astype(config.np_dtype)  # state cost matrix
R = np.identity(action_dim).astype(config.np_dtype)  # action cost matrix
K_lqr, P_lqr = mars.utils.lqr(A, B, Q, R)
print("LQR matrix:", K_lqr)
K = K_lqr

#2. NN control policy
controller_layer_dims = eval(args.controller_nn_sizes)
controller_layer_activations = eval(args.controller_nn_activations)
bound = 0.5
# policy = mars.NonLinearControllerLooseThresh(state_dim, controller_layer_dims,\
#     controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#     'high_slope':0.0, 'train_slope':args.controller_train_slope})
# policy = mars.NonLinearControllerLooseThreshWithLinearPart(state_dim, controller_layer_dims,\
#     -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
#     args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
#     'high_slope':0.0, 'train_slope':args.controller_train_slope})
policy = mars.NonLinearControllerLooseThreshWithLinearPartMulSlope(state_dim, controller_layer_dims,\
    -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
    args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
    'high_slope':0.0, 'train_slope':args.controller_train_slope, 'slope_multiplier':args.controller_slope_multiplier})

# Close loop dynamics with NN control policy
true_closed_loop_dynamics = lambda states: true_dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))
nominal_closed_loop_dynamics = lambda states: nominal_dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

# Initialize the Lyapunov functions ###################################################################
#1. Initialize the Quadratic Lyapunov function for the LQR controller and its induced ROA
L_pol = lambda x: np.linalg.norm(-K, 1) # # Policy (linear)
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) # Dynamics (linear approximation)
lyapunov_function = mars.QuadraticFunction(P_lqr)
grad_lyapunov_function = mars.LinearSystem((2 * P_lqr,))
#dot_v_lqr = lambda x: torch.sum(torch.mul(grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=1, keepdim=True) # Lipschitz constant of the Lyapunov function
L_dv = lambda x: torch.norm(torch.tensor(2 * P_lqr, dtype=config.ptdtype, device=device))
lyapunov_lqr = mars.Lyapunov_CT(grid, lyapunov_function, grad_lyapunov_function,\
     true_closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, L_v, L_dv, tau, initial_safe_set, decrease_thresh=0)
lyapunov_lqr.update_values()

#2.1 Initialize the NN Lyapunov Function
layer_dims = eval(args.roa_nn_sizes)
layer_activations = eval(args.roa_nn_activations)
decrease_thresh = args.lyapunov_decrease_threshold
# Initialize nn Lyapunov
L_pol = lambda x: np.linalg.norm(-K, 1) 
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) 
lyapunov_nn, grad_lyapunov_nn, dv_nn, L_v, tau = initialize_lyapunov_nn(grid, true_closed_loop_dynamics, None, L_dyn, 
            initial_safe_set, decrease_thresh, args.roa_nn_structure, state_dim, layer_dims, 
            layer_activations)
#####################################################################################################################

training_info = load_dict(os.path.join(results_dir, "training_info.npy"))
nominal_c_max_exp_values =  training_info["roa_info_nn"]["nominal_c_max_exp_values"]
nominal_c_max_exp_unconstrained_values =  training_info["roa_info_nn"]["nominal_c_max_exp_unconstrained_values"]

post_proc_info = {"grid_size":[], "roa_size":[], "forward_invariant_size":[],\
     "nn_forward_invariant_size":[], "lqr_forward_invariant_size":[],\
     "nn_roa_size":[], "lqr_roa_size":[]}

post_proc_info["nn_forward_invariant_size"] = copy.deepcopy(training_info["roa_info_nn"]["nominal_largest_exp_stable_set_sizes"])
post_proc_info["lqr_forward_invariant_size"] = copy.deepcopy(training_info["roa_info_lqr"]["nominal_largest_exp_stable_set_sizes"])
post_proc_info["nn_roa_size"] = copy.deepcopy(training_info["roa_info_nn"]["nominal_exp_stable_set_sizes"])
post_proc_info["lqr_roa_size"] = copy.deepcopy(training_info["roa_info_lqr"]["nominal_exp_stable_set_sizes"])

# Pretrained results
policy = load_controller_nn(policy, full_path=os.path.join(results_dir, 'init_controller_nn.net'))
lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'pretrained_lyapunov_nn.net'))
horizon = 4000 
tol = 0.01 
roa_true, trajs = compute_roa_ct(grid, true_closed_loop_dynamics, dt, horizon, tol, no_traj=False)
forward_invariant = np.zeros_like(roa_true, dtype=bool)
tmp = np.zeros([sum(roa_true),], dtype=bool)
trajs_abs = np.abs(trajs)[roa_true]
for i in range(trajs_abs.shape[0]):
    traj_abs = trajs_abs[i,:,:]
    if np.max(traj_abs) <= 1:
        tmp[i] = True
forward_invariant[roa_true] = tmp
print("ROA: ", sum(roa_true))
print("Forward invariance: ", sum(forward_invariant))


post_proc_info["grid_size"].append(grid.num_points)
post_proc_info["roa_size"].append(sum(roa_true))
post_proc_info["forward_invariant_size"].append(sum(forward_invariant))

for k in range(args.roa_outer_iters):
    print('Iteration {} out of {}'.format(k+1, args.roa_outer_iters))
    policy = load_controller_nn(policy, full_path=os.path.join(results_dir, 'trained_controller_nn_iter_{}.net'.format(k+1)))
    # lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'trained_lyapunov_nn_iter_{}.net'.format(k+1)))
    
    # Close loop dynamics and true region of attraction
    #closed_loop_dynamics = lambda states: true_dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

    roa_true, trajs = compute_roa_ct(grid, true_closed_loop_dynamics, dt, horizon, tol, no_traj=False)
    forward_invariant = np.zeros_like(roa_true, dtype=bool)
    tmp = np.zeros([sum(roa_true),], dtype=bool)
    trajs_abs = np.abs(trajs)[roa_true]
    for i in range(trajs_abs.shape[0]):
        traj_abs = trajs_abs[i,:,:]
        if np.max(traj_abs) <= 1:
            tmp[i] = True
    forward_invariant[roa_true] = tmp
    print("ROA: ", sum(roa_true))
    print("Forward invariance: ", sum(forward_invariant))
    post_proc_info["grid_size"].append(grid.num_points)
    post_proc_info["roa_size"].append(sum(roa_true))
    post_proc_info["forward_invariant_size"].append(sum(forward_invariant))

save_dict(post_proc_info, os.path.join(results_dir, "00post_proc_info.npy"))