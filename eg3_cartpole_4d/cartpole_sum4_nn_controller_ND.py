import os
import platform
# if platform.system() == 'Darwin':
#     os.environ['KMP_DUPLICATE_LIB_OK']='True'
#     print("KMP_DUPLICATE_LIB_OK set to true in os.environ to avoid warning on Mac")
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
from mars.visualization import plot_roa_3D, plot_nested_roas_3D
# from mars.roa_tools import initialize_roa, initialize_lyapunov_nn, initialize_lyapunov_quadratic, pretrain_lyapunov_nn, sample_around_roa, train_lyapunov_nn
from mars.roa_tools import *
from mars.dynamics_tools import *
from mars.utils import get_batch_grad, save_lyapunov_nn, load_lyapunov_nn,\
     save_dynamics_nn, load_dynamics_nn, save_controller_nn, load_controller_nn, count_parameters
from mars.controller_tools import pretrain_controller_nn, train_controller_SGD
from mars.parser_tools import getArgs

from examples.systems_config import all_systems 
from examples.example_utils import build_system, VanDerPol, InvertedPendulum, LyapunovNetwork, compute_roa_ct, balanced_class_weights, generate_trajectories, save_dict, load_dict


try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x
import pickle
import os
import warnings
import random
import argparse
warnings.filterwarnings("ignore")

input_args_str = "\
--system cartpole1\
--dt 0.01\
--drift_vector_nn_sizes [16,16,16,2]\
--drift_vector_nn_activations ['tanh','tanh','tanh','identity']\
--control_vector_nn_sizes [16,16,16,2]\
--control_vector_nn_activations ['tanh','tanh','tanh','identity']\
--dynamics_batchsize 64\
--dynamics_loss_coeff 1e4\
--dynamics_pre_lr 1e-3\
--dynamics_pre_iters 10\
--dynamics_train_lr 1e-3\
--dynamics_train_iters 10\
--grid_resolution 10\
--repetition_use denoise\
--lyapunov_decrease_threshold 0.0\
--roa_gridsize 100\
--roa_pre_lr 1e-3\
--roa_pre_iters 30\
--roa_pre_batchsize 64\
--roa_inner_iters 10\
--roa_outer_iters 300\
--roa_train_lr 5e-6\
--roa_lr_scheduler_step 40\
--roa_nn_structure sum_of_two_eth\
--roa_nn_sizes [64,64,64]\
--roa_nn_activations ['tanh','tanh','tanh']\
--roa_batchsize 64\
--roa_adaptive_level_multiplier False\
--roa_adaptive_level_multiplier_step 50\
--roa_level_multiplier 10\
--roa_decrease_loss_coeff 500.0\
--roa_decrease_alpha 0.1\
--roa_decrease_offset 0.01\
--roa_lipschitz_loss_coeff 0.01\
--roa_c_target 0.04\
--roa_classification_loss_coeff 2\
--controller_nn_sizes [16,16,16,1]\
--controller_nn_activations ['tanh','tanh','tanh','identity']\
--controller_slope_multiplier 50\
--controller_pre_lr 1e-2\
--controller_pre_iters 500\
--controller_pre_batchsize 64\
--controller_inner_iters 10\
--controller_outer_iters 2\
--controller_level_multiplier 2\
--controller_traj_length 10\
--controller_train_lr 5e-6\
--controller_batchsize 16\
--controller_train_slope True\
--verbose True\
--image_save_format pdf\
--exp_num 800\
--use_cuda False\
--cutoff_radius 0.4"


input_args_temp = input_args_str.split("--")
input_args = []
for ind, twins in enumerate(input_args_temp[1:]):
    a, b = twins.split(" ")
    a = "--{}".format(a)
    input_args.append(a)
    input_args.append(b)
args = getArgs(input_args)

device = config.device
print('Pytorch using device:', device)
exp_num = args.exp_num
results_dir = '{}/results/exp_{:03d}'.format(str(Path(__file__).parent.parent), exp_num)
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

# Save hyper-parameters
with open(os.path.join(results_dir, "00hyper_parameters.txt"), "w") as text_file:
    input_args_temp = input_args_str.split("--")
    for i in range(1, len(input_args_temp)):
        text_file.write("--")
        text_file.write(str(input_args_temp[i]))
        text_file.write("\\")
        text_file.write("\n")

# Set random seed
seed_num = 0
torch.manual_seed(seed_num)
np.random.seed(seed_num)
# Choosing the system
dt = args.dt   # sampling time
# System properties
system_properties = all_systems[args.system]
nominal_system_properties = all_systems[args.system + '_nominal']
state_dim     = system_properties.state_dim
action_dim    = system_properties.action_dim
state_limits  = np.array([[-1., 1.]] * state_dim)
action_limits = np.array([[-1., 1.]] * action_dim)
resolution = args.grid_resolution # Number of states divisions each dimension

# Save system config params
with open(os.path.join(results_dir, "00system_config.txt"), "w") as text_file:
    text_file.write("True system parameters: \n")
    true_sys = vars(system_properties)
    for key in true_sys:
        text_file.write(str(key))
        text_file.write(" = ")
        text_file.write(str(true_sys[key]))
        text_file.write("\n")
    text_file.write("\n")
    text_file.write("Nominal system parameters: \n")
    nom_sys = vars(nominal_system_properties)
    for key in nom_sys:
        text_file.write(str(key))
        text_file.write(" = ")
        text_file.write(str(nom_sys[key]))
        text_file.write("\n")

# Initialize system class and its linearization
system = build_system(system_properties, dt)
nominal_system = build_system(nominal_system_properties, dt)
A, B = nominal_system.linearize_ct()
# A, B = system.linearize_ct()
dynamics = lambda x, y: system.ode_normalized(x, y)
nominal_dynamics = lambda x, y: nominal_system.ode_normalized(x, y)

# State grid
grid_limits = np.array([[-1., 1.], ] * state_dim)
grid = mars.GridWorld(grid_limits, resolution)
tau = np.sum(grid.unit_maxes) / 2
u_max = system.normalization[1].item()
Tx, Tu = map(np.diag, system.normalization)
Tx_inv, Tu_inv = map(np.diag, system.inv_norm)

# Set initial safe set as a ball around the origin (in normalized coordinates)
cutoff_radius    = args.cutoff_radius
initial_safe_set = np.linalg.norm(grid.all_points, ord=2, axis=1) <= cutoff_radius

# LQR policy and its true ROA
Q = np.identity(state_dim).astype(config.np_dtype)  # state cost matrix
R = np.identity(action_dim).astype(config.np_dtype)  # action cost matrix
K_lqr, P_lqr = mars.utils.lqr(A, B, Q, R)
print("LQR matrix:", K_lqr)
K = K_lqr

controller_layer_dims = eval(args.controller_nn_sizes)
controller_layer_activations = eval(args.controller_nn_activations)

bound = 0.5
policy = mars.NonLinearControllerLooseThreshWithLinearPartMulSlope(state_dim, controller_layer_dims,\
    -K, controller_layer_activations, initializer=torch.nn.init.xavier_uniform,\
    args={'low_thresh':-bound, 'high_thresh':bound, 'low_slope':0.0, \
    'high_slope':0.0, 'train_slope':args.controller_train_slope, 'slope_multiplier':args.controller_slope_multiplier})

save_controller_nn(policy, full_path=os.path.join(results_dir, 'init_controller_nn.net'))

print("Policy:", policy.mul_low_slope_param, policy.mul_high_slope_param)
print("Trainable Parameters (policy): ", count_parameters(policy))

# print('Load previous weights for controller network')
# full_path=os.path.join(results_dir, 'pretrained_controller_nn.net')
# policy = load_controller_nn(policy, full_path)

# Close loop dynamics
closed_loop_dynamics = lambda states: dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))
partial_closed_loop_dynamics = lambda states: nominal_dynamics(torch.tensor(states, device = device), policy(torch.tensor(states, device = device)))

horizon = 2500 # smaller tol requires longer horizon to give an accurate estimate of ROA
tol = 0.01 # how much close to origin must be x(T) to be considered as stable trajectory
roa_true = compute_roa_ct(grid, closed_loop_dynamics, dt, horizon, tol, no_traj=True) # True ROA with LQR policy
print("Size of ROA init:", np.sum(roa_true))


nominal_closed_loop_dynamics = None
# Quadratic Lyapunov function for the LQR controller and its induced ROA
L_pol = lambda x: np.linalg.norm(-K, 1) # # Policy (linear)
L_dyn = lambda x: np.linalg.norm(A, 1) + np.linalg.norm(B, 1) * L_pol(x) # Dynamics (linear approximation)
lyapunov_function = mars.QuadraticFunction(P_lqr)
grad_lyapunov_function = mars.LinearSystem((2 * P_lqr,))
dot_v_lqr = lambda x: torch.sum(torch.mul(grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
L_v = lambda x: torch.norm(grad_lyapunov_function(x), p=1, dim=1, keepdim=True) # Lipschitz constant of the Lyapunov function
L_dv = lambda x: torch.norm(torch.tensor(2 * P_lqr, dtype=config.ptdtype, device=device))
lyapunov_lqr = mars.Lyapunov_CT(grid, lyapunov_function, grad_lyapunov_function,\
     closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, L_v, L_dv, tau, initial_safe_set, decrease_thresh=0)
lyapunov_lqr.update_values()
lyapunov_lqr.update_safe_set('true', roa_true)
lyapunov_lqr.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)

print("Stable states: {}".format(roa_true.sum()))
print("Size of initial_safe_set: ", initial_safe_set.sum())
print("Largest ROA contained (LQR): {}".format(lyapunov_lqr.largest_safe_set_true.sum()))
print("Largest exp stable contained (LQR): {}".format(lyapunov_lqr.largest_exp_stable_set_true.sum()))
print("ROA size (LQR): {}".format(lyapunov_lqr.safe_set_true.sum()))
print("Exp stable size (LQR): {}".format(lyapunov_lqr.exp_stable_set_true.sum()))


###### NN Lyapunov ######
layer_dims = eval(args.roa_nn_sizes)
layer_activations = eval(args.roa_nn_activations)
decrease_thresh = args.lyapunov_decrease_threshold
# Initialize nn Lyapunov
lyapunov_nn, grad_lyapunov_nn, dv_nn, L_v, tau = initialize_lyapunov_nn(grid, closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, 
            initial_safe_set, decrease_thresh, args.roa_nn_structure, state_dim, layer_dims, 
            layer_activations)

lyapunov_nn.update_values()
lyapunov_nn.update_safe_set('true', roa_true)
lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha,'true', roa_true)
save_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'untrained_lyapunov_nn.net'))

print("Stable states: {}".format(roa_true.sum()))
print("Size of initial_safe_set: ", initial_safe_set.sum())
print("Largest ROA contained (NN init): {}".format(lyapunov_nn.largest_safe_set_true.sum()))
print("Largest exp stable contained (NN init): {}".format(lyapunov_nn.largest_exp_stable_set_true.sum()))
print("ROA size (NN init): {}".format(lyapunov_nn.safe_set_true.sum()))
print("Exp stable size (NN init): {}".format(lyapunov_nn.exp_stable_set_true.sum()))



################ Pretrain to a circle ##############
# Initialize quadratic Lyapunov
P = 0.1 * np.eye(state_dim)
lyapunov_pre, grad_lyapunov_pre, L_v_pre, L_dv_pre, tau = initialize_lyapunov_quadratic(grid, P, closed_loop_dynamics, nominal_closed_loop_dynamics, L_dyn, 
                                                    initial_safe_set, decrease_thresh)
lyapunov_pre.update_values()
lyapunov_pre.update_safe_set('true', roa_true)
lyapunov_pre.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)

print("Stable states: {}".format(roa_true.sum()))
print("Size of initial_safe_set: ", initial_safe_set.sum())
print("Largest ROA contained (pre target): {}".format(lyapunov_pre.largest_safe_set_true.sum()))
print("Largest exp stable contained (pre target): {}".format(lyapunov_pre.largest_exp_stable_set_true.sum()))
print("ROA size (pre target): {}".format(lyapunov_pre.safe_set_true.sum()))
print("Exp stable size (pre target): {}".format(lyapunov_pre.exp_stable_set_true.sum()))


print("Pretrain the Lyapunov network:")
pretrain_lyapunov_nn_Adam(grid, lyapunov_nn, lyapunov_pre, args.roa_pre_batchsize, args.roa_pre_iters, args.roa_pre_lr,\
    verbose=False, full_path = os.path.join(results_dir, 'roa_training_loss_pretrain.{}'.format(args.image_save_format)))
save_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'pretrained_lyapunov_nn.net'))

# print('Load previous weights for Lyapunov network')
# full_path=os.path.join(results_dir, 'pretrained_lyapunov_nn.net')
# lyapunov_nn = load_lyapunov_nn(lyapunov_nn, full_path)

lyapunov_nn.update_values()
lyapunov_nn.update_safe_set('true', roa_true)
lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)

print("Stable states: {}".format(roa_true.sum()))
print("Size of initial_safe_set: ", initial_safe_set.sum())
print("Largest ROA contained (NN pretrained): {}".format(lyapunov_nn.largest_safe_set_true.sum()))
print("Largest exp stable contained (NN pretrained): {}".format(lyapunov_nn.largest_exp_stable_set_true.sum()))
print("ROA size (NN pretrained): {}".format(lyapunov_nn.safe_set_true.sum()))
print("Exp stable size (NN pretrained): {}".format(lyapunov_nn.exp_stable_set_true.sum()))


# Create error nets
layer_dims = eval(args.control_vector_nn_sizes)
layer_activations = eval(args.control_vector_nn_activations)
control_vec_nn = mars.DynamicsNet(state_dim, layer_dims, layer_activations, initializer=torch.nn.init.xavier_uniform).to(device)
layer_dims = eval(args.drift_vector_nn_sizes)
layer_activations = eval(args.drift_vector_nn_activations)
drift_vec_nn = mars.DynamicsNet(state_dim, layer_dims, layer_activations, initializer=torch.nn.init.xavier_uniform).to(device)
nominal_closed_loop_dynamics = lambda states: partial_closed_loop_dynamics(states) \
    + torch.cat((torch.zeros([len(states),2], dtype = config.ptdtype, device=device), drift_vec_nn(states)), dim=1)\
    + torch.cat((torch.zeros([len(states),2], dtype = config.ptdtype, device=device), \
        torch.mul(control_vec_nn(states), policy(torch.tensor(states, device=device)))), dim=1)

sampling_time = 0.001
dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), closed_loop_dynamics(x)),1)
dot_vnn_nd = dot_vnn # TODO: temporary code for testing
#dot_vnn_nd = lambda x: torch.squeeze((lyapunov_nn.lyapunov_function(torch.tensor(x, dtype = config.ptdtype, device=device)\
#    + closed_loop_dynamics(x)*sampling_time) - lyapunov_nn.lyapunov_function(x))/sampling_time)
nominal_dot_vnn = lambda x: torch.sum(torch.mul(lyapunov_nn.grad_lyapunov_function(x), nominal_closed_loop_dynamics(x)),1)


print("Pretrain the dynamics error network:")
target_idx = lyapunov_nn.values.detach().cpu().numpy().ravel() <= lyapunov_nn.c_max_exp_true*(args.roa_level_multiplier +1)
target_set = grid.all_points[target_idx]
train_dynamics_sample_in_batch_Adam(target_set, dot_vnn_nd, nominal_dot_vnn, drift_vec_nn, control_vec_nn,\
        args.dynamics_loss_coeff, args.dynamics_batchsize, args.dynamics_pre_iters, args.dynamics_pre_lr, \
        full_path=os.path.join(results_dir, 'dynamics_training_loss_pretrain.{}'.format(args.image_save_format)))
save_dynamics_nn(drift_vec_nn, full_path=os.path.join(results_dir, 'pretrained_drift_vec_nn.net'))
save_dynamics_nn(control_vec_nn, full_path=os.path.join(results_dir, 'pretrained_control_vec_nn.net'))


# ## Or load previous weights
# print('Load previous weights for dynamics error network')
# drift_vec_nn = load_dynamics_nn(drift_vec_nn, full_path=os.path.join(results_dir, 'pretrained_drift_vec_nn.net'))
# control_vec_nn = load_dynamics_nn(control_vec_nn, full_path=os.path.join(results_dir, 'pretrained_control_vec_nn.net'))


lyapunov_nn.closed_loop_dynamics_nominal = nominal_closed_loop_dynamics
lyapunov_lqr.closed_loop_dynamics_nominal = nominal_closed_loop_dynamics
lyapunov_nn.update_values()
lyapunov_nn.update_safe_set('nominal', roa_true)
lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha, 'nominal', roa_true)
lyapunov_lqr.update_values()
lyapunov_lqr.update_safe_set('nominal', roa_true)
lyapunov_lqr.update_exp_stable_set(args.roa_decrease_alpha, 'nominal', roa_true)


# Monitor the training process
training_info = {"grid_size":{}, "roa_info_nn":{}, "roa_info_lqr":{}, "policy_info":{}}
training_info["grid_size"] = grid.nindex
training_info["roa_info_nn"] = {"true_roa_sizes": [],\
    "true_largest_safe_set_sizes": [], "nominal_largest_safe_set_sizes":[],\
    "true_safe_set_sizes": [], "nominal_safe_set_sizes":[],\
    "true_c_max_values": [], "nominal_c_max_values": [],\
    "true_c_max_unconstrained_values": [], "nominal_c_max_unconstrained_values": [],\
    "true_largest_exp_stable_set_sizes": [], "nominal_largest_exp_stable_set_sizes":[],\
    "true_exp_stable_set_sizes": [], "nominal_exp_stable_set_sizes":[],\
    "true_c_max_exp_values": [], "nominal_c_max_exp_values": [],\
    "true_c_max_exp_unconstrained_values": [], "nominal_c_max_exp_unconstrained_values": []}
training_info["roa_info_lqr"] = {"true_roa_sizes": [],\
    "true_largest_safe_set_sizes": [], "nominal_largest_safe_set_sizes":[],\
    "true_safe_set_sizes": [], "nominal_safe_set_sizes":[],\
    "true_c_max_values": [], "nominal_c_max_values": [],\
    "true_c_max_unconstrained_values": [], "nominal_c_max_unconstrained_values": [],\
    "true_largest_exp_stable_set_sizes": [], "nominal_largest_exp_stable_set_sizes":[],\
    "true_exp_stable_set_sizes": [], "nominal_exp_stable_set_sizes":[],\
    "true_c_max_exp_values": [], "nominal_c_max_exp_values": [],\
    "true_c_max_exp_unconstrained_values": [], "nominal_c_max_exp_unconstrained_values": []}
training_info["policy_info"] = {"low_thresh_param" : [], "high_thresh_param" : [], "low_slope_param" : [], "high_slope_param" : []}

# Approximate ROA
print("Policy:", policy.mul_low_slope_param, policy.mul_high_slope_param)
print("Stable states: {}".format(roa_true.sum()))
print("Largest ROA contained: {}".format(lyapunov_nn.largest_safe_set_true.sum()))
print("Largest exp stable contained: {}".format(lyapunov_nn.largest_exp_stable_set_true.sum()))
print("ROA size: {}".format(lyapunov_nn.safe_set_true.sum()))
print("Exp stable size: {}".format(lyapunov_nn.exp_stable_set_true.sum()))

training_info["policy_info"]["low_thresh_param"].append(copy.deepcopy(policy.low_thresh_param.detach().cpu().numpy()))
training_info["policy_info"]["high_thresh_param"].append(copy.deepcopy(policy.high_thresh_param.detach().cpu().numpy()))
training_info["policy_info"]["low_slope_param"].append(copy.deepcopy(policy.mul_low_slope_param.detach().cpu().numpy()))
training_info["policy_info"]["high_slope_param"].append(copy.deepcopy(policy.mul_high_slope_param.detach().cpu().numpy()))

training_info["roa_info_nn"]["true_roa_sizes"].append(sum(roa_true))
training_info["roa_info_nn"]["true_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.safe_set_true)))
training_info["roa_info_nn"]["true_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_safe_set_true)))
training_info["roa_info_nn"]["true_c_max_values"].append(copy.deepcopy(lyapunov_nn.c_max_true))
training_info["roa_info_nn"]["true_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_unconstrained_true))
training_info["roa_info_nn"]["true_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.exp_stable_set_true)))
training_info["roa_info_nn"]["true_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_exp_stable_set_true)))
training_info["roa_info_nn"]["true_c_max_exp_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_true))
training_info["roa_info_nn"]["true_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_unconstrained_true))
training_info["roa_info_nn"]["nominal_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.safe_set_nominal)))
training_info["roa_info_nn"]["nominal_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_safe_set_nominal)))
training_info["roa_info_nn"]["nominal_c_max_values"].append(copy.deepcopy(lyapunov_nn.c_max_nominal))
training_info["roa_info_nn"]["nominal_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_unconstrained_nominal))
training_info["roa_info_nn"]["nominal_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.exp_stable_set_nominal)))
training_info["roa_info_nn"]["nominal_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_exp_stable_set_nominal)))
training_info["roa_info_nn"]["nominal_c_max_exp_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_nominal))
training_info["roa_info_nn"]["nominal_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_unconstrained_nominal))

training_info["roa_info_lqr"]["true_roa_sizes"].append(sum(roa_true))
training_info["roa_info_lqr"]["true_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.safe_set_true)))
training_info["roa_info_lqr"]["true_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_safe_set_true)))
training_info["roa_info_lqr"]["true_c_max_values"].append(copy.deepcopy(lyapunov_lqr.c_max_true))
training_info["roa_info_lqr"]["true_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_unconstrained_true))
training_info["roa_info_lqr"]["true_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.exp_stable_set_true)))
training_info["roa_info_lqr"]["true_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_exp_stable_set_true)))
training_info["roa_info_lqr"]["true_c_max_exp_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_true))
training_info["roa_info_lqr"]["true_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_unconstrained_true))
training_info["roa_info_lqr"]["nominal_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.safe_set_nominal)))
training_info["roa_info_lqr"]["nominal_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_safe_set_nominal)))
training_info["roa_info_lqr"]["nominal_c_max_values"].append(copy.deepcopy(lyapunov_lqr.c_max_nominal))
training_info["roa_info_lqr"]["nominal_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_unconstrained_nominal))
training_info["roa_info_lqr"]["nominal_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.exp_stable_set_nominal)))
training_info["roa_info_lqr"]["nominal_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_exp_stable_set_nominal)))
training_info["roa_info_lqr"]["nominal_c_max_exp_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_nominal))
training_info["roa_info_lqr"]["nominal_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_unconstrained_nominal))


c = lyapunov_nn.c_max_exp_nominal

idx_small = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c
if args.roa_adaptive_level_multiplier:
    level_multiplier = 1 + args.roa_level_multiplier
else: 
    level_multiplier = args.roa_level_multiplier
idx_big = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c * level_multiplier
idx_exp_stable_true = lyapunov_nn.exp_stable_set_true


gamma = 0.5
lr_multiplier = lambda k : gamma**(k//args.roa_lr_scheduler_step)

t_start = perf_counter()
for k in range(args.roa_outer_iters):
    t_epoch_start = perf_counter()
    print('Iteration {} out of {}'.format(k+1, args.roa_outer_iters))
    print('Level multiplier: {}'.format(level_multiplier))
    if lr_multiplier != None:
        roa_lr = args.roa_train_lr*lr_multiplier(k)
        controller_lr = args.controller_train_lr*lr_multiplier(k)
        print('Learning rates: ', roa_lr, controller_lr)
    # target_idx = np.logical_and(idx_big, np.logical_not(initial_safe_set))
    target_idx = np.logical_or(idx_big, initial_safe_set)
    print('Size of training set: ', np.sum(target_idx))
    if np.sum(target_idx) == 0:
        print("No stable points in the training set! Abort training!")
        break
    target_set = grid.all_points[target_idx]
    train_dynamics_sample_in_batch_Adam(target_set, dot_vnn_nd, nominal_dot_vnn, drift_vec_nn, control_vec_nn,\
        args.dynamics_loss_coeff, args.dynamics_batchsize, args.dynamics_train_iters, args.dynamics_train_lr, \
        full_path=os.path.join(results_dir, 'dynamics_training_loss_iter_{}.{}'.format(k+1, args.image_save_format)),
        print_grad=False)
    save_dynamics_nn(drift_vec_nn, full_path=os.path.join(results_dir, 'trained_drift_vec_nn_iter_{}.net'.format(k+1)))
    save_dynamics_nn(control_vec_nn, full_path=os.path.join(results_dir, 'trained_control_vec_nn_iter_{}.net'.format(k+1)))
    train_largest_ROA_Adam(target_set, lyapunov_nn, policy, nominal_closed_loop_dynamics, args.roa_batchsize,
        args.roa_inner_iters, roa_lr, controller_lr,
        args.roa_decrease_alpha, args.roa_decrease_offset, args.roa_size_beta,
        args.roa_decrease_loss_coeff, args.roa_lipschitz_loss_coeff, args.roa_size_loss_coeff,
        fullpath_to_save_objectives=os.path.join(results_dir, 'roa_training_loss_iter_{}.{}'.format(k+1, args.image_save_format)),
        verbose=False, optimizer = None, lr_scheduler=None)
    save_lyapunov_nn(lyapunov_nn, full_path=os.path.join(results_dir, 'trained_lyapunov_nn_iter_{}.net'.format(k+1)))
    save_controller_nn(policy, full_path=os.path.join(results_dir, 'trained_controller_nn_iter_{}.net'.format(k+1)))
    print("Policy:", policy.mul_low_slope_param, policy.mul_high_slope_param)
    
    horizon = 4000
    roa_true = compute_roa_ct(grid, closed_loop_dynamics, dt, horizon, tol, no_traj=True) # True ROA with LQR policy
    
    lyapunov_nn.update_values()
    lyapunov_nn.update_safe_set('true', roa_true)
    lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)
    lyapunov_nn.update_safe_set('nominal', roa_true)
    lyapunov_nn.update_exp_stable_set(args.roa_decrease_alpha, 'nominal', roa_true)

    lyapunov_lqr.update_values()
    lyapunov_lqr.update_safe_set('true', roa_true)
    lyapunov_lqr.update_exp_stable_set(args.roa_decrease_alpha, 'true', roa_true)
    lyapunov_lqr.update_safe_set('nominal', roa_true)
    lyapunov_lqr.update_exp_stable_set(args.roa_decrease_alpha, 'nominal', roa_true)

    training_info["policy_info"]["low_thresh_param"].append(copy.deepcopy(policy.low_thresh_param.detach().cpu().numpy()))
    training_info["policy_info"]["high_thresh_param"].append(copy.deepcopy(policy.high_thresh_param.detach().cpu().numpy()))
    training_info["policy_info"]["low_slope_param"].append(copy.deepcopy(policy.mul_low_slope_param.detach().cpu().numpy()))
    training_info["policy_info"]["high_slope_param"].append(copy.deepcopy(policy.mul_high_slope_param.detach().cpu().numpy()))

    training_info["roa_info_nn"]["true_roa_sizes"].append(sum(roa_true))
    training_info["roa_info_nn"]["true_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.safe_set_true)))
    training_info["roa_info_nn"]["true_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_safe_set_true)))
    training_info["roa_info_nn"]["true_c_max_values"].append(copy.deepcopy(lyapunov_nn.c_max_true))
    training_info["roa_info_nn"]["true_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_unconstrained_true))
    training_info["roa_info_nn"]["true_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.exp_stable_set_true)))
    training_info["roa_info_nn"]["true_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_exp_stable_set_true)))
    training_info["roa_info_nn"]["true_c_max_exp_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_true))
    training_info["roa_info_nn"]["true_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_unconstrained_true))
    training_info["roa_info_nn"]["nominal_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.safe_set_nominal)))
    training_info["roa_info_nn"]["nominal_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_safe_set_nominal)))
    training_info["roa_info_nn"]["nominal_c_max_values"].append(copy.deepcopy(lyapunov_nn.c_max_nominal))
    training_info["roa_info_nn"]["nominal_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_unconstrained_nominal))
    training_info["roa_info_nn"]["nominal_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.exp_stable_set_nominal)))
    training_info["roa_info_nn"]["nominal_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_nn.largest_exp_stable_set_nominal)))
    training_info["roa_info_nn"]["nominal_c_max_exp_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_nominal))
    training_info["roa_info_nn"]["nominal_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_nn.c_max_exp_unconstrained_nominal))

    training_info["roa_info_lqr"]["true_roa_sizes"].append(sum(roa_true))
    training_info["roa_info_lqr"]["true_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.safe_set_true)))
    training_info["roa_info_lqr"]["true_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_safe_set_true)))
    training_info["roa_info_lqr"]["true_c_max_values"].append(copy.deepcopy(lyapunov_lqr.c_max_true))
    training_info["roa_info_lqr"]["true_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_unconstrained_true))
    training_info["roa_info_lqr"]["true_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.exp_stable_set_true)))
    training_info["roa_info_lqr"]["true_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_exp_stable_set_true)))
    training_info["roa_info_lqr"]["true_c_max_exp_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_true))
    training_info["roa_info_lqr"]["true_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_unconstrained_true))
    training_info["roa_info_lqr"]["nominal_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.safe_set_nominal)))
    training_info["roa_info_lqr"]["nominal_largest_safe_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_safe_set_nominal)))
    training_info["roa_info_lqr"]["nominal_c_max_values"].append(copy.deepcopy(lyapunov_lqr.c_max_nominal))
    training_info["roa_info_lqr"]["nominal_c_max_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_unconstrained_nominal))
    training_info["roa_info_lqr"]["nominal_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.exp_stable_set_nominal)))
    training_info["roa_info_lqr"]["nominal_largest_exp_stable_set_sizes"].append(copy.deepcopy(sum(lyapunov_lqr.largest_exp_stable_set_nominal)))
    training_info["roa_info_lqr"]["nominal_c_max_exp_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_nominal))
    training_info["roa_info_lqr"]["nominal_c_max_exp_unconstrained_values"].append(copy.deepcopy(lyapunov_lqr.c_max_exp_unconstrained_nominal))

    print("Stable states: {}".format(roa_true.sum()))
    c_true = lyapunov_nn.c_max_exp_true
    print("c_max_exp_true: {}".format(c_true))
    c = lyapunov_nn.c_max_exp_nominal
    print("c_max_exp_nominal: {}".format(c))
    print("Largest ROA contained: {}".format(lyapunov_nn.largest_safe_set_true.sum()))
    print("Largest exp stable contained: {}".format(lyapunov_nn.largest_exp_stable_set_true.sum()))
    print("ROA size: {}".format(lyapunov_nn.safe_set_true.sum()))
    print("Exp stable size: {}".format(lyapunov_nn.exp_stable_set_true.sum()))
    idx_small = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c
    if args.roa_adaptive_level_multiplier:
        level_multiplier = 1 + args.roa_level_multiplier/((k//args.roa_adaptive_level_multiplier_step)+1)
    else: 
        level_multiplier = args.roa_level_multiplier
    idx_big = lyapunov_nn.values.detach().cpu().numpy().ravel() <= c * level_multiplier
    idx_exp_stable = lyapunov_nn.exp_stable_set_true
    t_epoch_stop = perf_counter()
    print("Elapsed time during iteration {} in seconds:".format(k+1), t_epoch_stop-t_epoch_start)

t_stop = perf_counter()
print("Elapsed time during training and plotting in seconds:", t_stop-t_start)
save_dict(training_info, os.path.join(results_dir, "training_info.npy"))

