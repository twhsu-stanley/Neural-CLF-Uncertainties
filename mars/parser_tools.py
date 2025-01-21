import argparse
from .utils import str2bool

def getArgs(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', default='pendulum1', type=str, help='pick a dynamical system')
    parser.add_argument('--dt', default=0.01, type=float, help='time interval of simulations')
    parser.add_argument('--drift_vector_nn_sizes', default='[16, 16, 32]', type=str, help='number of neurons at each layer of the neural network of ROA')
    parser.add_argument('--drift_vector_nn_activations', default="['relu', 'relu', 'relu']", type=str, help='the activation of each layer of the neural network of dynamics')
    parser.add_argument('--control_vector_nn_sizes', default='[16, 16, 32]', type=str, help='number of neurons at each layer of the neural network of ROA')
    parser.add_argument('--control_vector_nn_activations', default="['relu', 'relu', 'relu']", type=str, help='the activation of each layer of the neural network of dynamics')
    parser.add_argument('--dynamics_loss_coeff', default=100, type=float, help='the weight of the decrease condition in the total objective')
    parser.add_argument('--dynamics_batchsize', default=64, type=int, help='the batchsize to train the neural network of ROA')
    parser.add_argument('--dynamics_pre_lr', default=0.001, type=float, help='learning rate to pretrain the neural network of ROA')
    parser.add_argument('--dynamics_pre_iters', default=100, type=int, help='number of iterations to train the neural network of ROA at each growth stage')
    parser.add_argument('--dynamics_train_lr', default=0.001, type=float, help='learning rate to pretrain the neural network of ROA')
    parser.add_argument('--dynamics_train_iters', default=100, type=int, help='number of iterations to train the neural network of ROA at each growth stage')
    parser.add_argument('--grid_resolution', default=100, type=int, help='number of division in every dimension of the state grid')
    parser.add_argument('--repetition_use', default='denoise', type=str, help='how to use the repititions in each environment {denoise,learn}')
    parser.add_argument('--roa_gridsize', default=100, type=int, help='number of discrerized states in each dimension of the ROA grid')
    parser.add_argument('--roa_pre_lr', default=0.001, type=float, help='learning rate to pretrain the neural network of ROA')
    parser.add_argument('--roa_pre_iters', default=10000, type=int, help='number of iterations to pretrain the neural network of ROA')
    parser.add_argument('--roa_pre_batchsize', default=16, type=int, help='the batchsize to train the neural network of ROA')
    parser.add_argument('--roa_batchsize', default=16, type=int, help='the batchsize to train the neural network of ROA')
    parser.add_argument('--roa_train_lr', default=0.1, type=float, help='learning rate to train the neural network of ROA while growing')
    parser.add_argument('--roa_lr_scheduler_step', default=20, type=int, help='learning rate to train the neural network of ROA while growing')
    parser.add_argument('--roa_lr_scheduler_gamma', default=0.5, type=float, help='learning rate to train the neural network of ROA while growing')
    parser.add_argument('--roa_inner_iters', default=100, type=int, help='number of iterations to train the neural network of ROA at each growth stage')
    parser.add_argument('--roa_outer_iters', default=100, type=int, help='number of growth stages to train the neural network of ROA')
    parser.add_argument('--roa_adaptive_level_multiplier', default=True, type=str2bool, help='gradually decrease level multiplier')
    parser.add_argument('--roa_adaptive_level_multiplier_step', default=4, type=int, help='learning rate to train the neural network of ROA while growing')
    parser.add_argument('--roa_level_multiplier', default=3.0, type=float, help='determines how much the levelset grows at each growth stage (> 1.0)')
    parser.add_argument('--roa_decrease_loss_coeff', default=200, type=float, help='the weight of the decrease condition in the total objective')
    parser.add_argument('--roa_decrease_alpha', default=0.1, type=float, help='the weight of the decrease condition in the total objective')
    parser.add_argument('--roa_decrease_offset', default=0.0, type=float, help='the weight of the decrease condition in the total objective')
    parser.add_argument('--roa_size_beta', default=0.1, type=float, help='the weight of the decrease condition in the total objective')
    parser.add_argument('--roa_lipschitz_loss_coeff', default=0.000, type=float, help='the weight of the Lipschitz condition in the total objective')
    parser.add_argument('--roa_size_loss_coeff', default=0.000, type=float, help='the weight of the roa size loss in the total objective')
    parser.add_argument('--roa_c_target', default=1.0, type=float, help='the target c')
    parser.add_argument('--roa_classification_loss_coeff', default=1.0, type=float, help='the weight of the classification loss in the total objective')
    parser.add_argument('--roa_nn_structure', default='quadratic', type=str, help='structure of the neural network of ROA')
    parser.add_argument('--roa_nn_sizes', default='[16, 16, 32]', type=str, help='number of neurons at each layer of the neural network of ROA')
    parser.add_argument('--roa_nn_activations', default="['tanh', 'tanh', 'tanh']", type=str, help='the activation of each layer of the neural network of ROA')
    parser.add_argument('--lyapunov_decrease_threshold', default=0.0, type=float, help='considered as satisfied if less than this threshold (must be negative in theory)')
    parser.add_argument('--controller_nn_sizes', default='[16, 16, 32]', type=str, help='number of neurons at each layer of the neural network of controller')
    parser.add_argument('--controller_nn_activations', default="['tanh', 'tanh', 'tanh']", type=str, help='the activation of each layer of the neural network of controller')
    parser.add_argument('--controller_slope_multiplier', default=100, type=float, help='learning rate to pretrain the neural network of controller')
    parser.add_argument('--controller_pre_lr', default=0.001, type=float, help='learning rate to pretrain the neural network of controller')
    parser.add_argument('--controller_pre_iters', default=10000, type=int, help='number of iterations to pretrain the neural network of controller')
    parser.add_argument('--controller_pre_batchsize', default=16, type=int, help='the batchsize to train the neural network of controller')
    parser.add_argument('--controller_outer_iters', default=10, type=int, help='number of times that a new controller is discovered')
    parser.add_argument('--controller_inner_iters', default=1000, type=int, help='number of iterations to train the controller at each growth stage')
    parser.add_argument('--controller_level_multiplier', default=3.0, type=float, help='determines how much the levelset grows at each growth stage (> 1.0)')
    parser.add_argument('--controller_traj_length', default=100, type=int, help='the length of the trajectory produced to train the controller')
    parser.add_argument('--controller_train_lr', default=0.01, type=float, help='learning rate to train the controller')
    parser.add_argument('--controller_batchsize', default=16, type=int, help='the batchsize to train the controller')
    parser.add_argument('--controller_train_slope', default=False, type=str2bool, help='train slope of controller or not')
    parser.add_argument('--verbose', default=False, type=str2bool, help='print out the state of the learners')
    parser.add_argument('--use_cuda', default=True, type=str2bool, help='Use GPU or CPU')
    parser.add_argument('--exp_num', default=0, type=int, help='the number of the experiment. It determines the folder in which the results are saved or loaded from.')
    parser.add_argument('--image_save_format', default="png", type=str, help='png or pdf')
    parser.add_argument('--image_save_format_3d', default="png", type=str, help='png or pdf')
    parser.add_argument('--cutoff_radius', default=0.1, type=float, help='learning rate to train the controller')
    parser.add_argument('--use_cp', default=False, type=str2bool, help='Use conformal prediction or not')
    parser.add_argument('--roa_decrease_loss_cp_coeff', default=100, type=float, help='the weight of the decrease condition with cp in the total objective')
    return parser.parse_args(argv)
