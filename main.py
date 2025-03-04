# %%
# Load packages
# import torch_funcs at start of program to set default type before main()
from utils.plots.plot_multiple_fields import plot_multiple_fields
from model.postprocessing import postprocessing
from utils.class_welfords_online_algorithm import WelfordsOnlineAlgorithm
from utils.function_copy_input_file import copy_input_file
from utils.function_create_result_dict import create_result_dict
from model.convergence_criterion import convergence_criterion
from model.class_log import log
from model.class_SVI import SVI
from model.Posteriors.Posterior_selection import posterior_selection_X, posterior_selection_Y
from model.ParentClasses.class_Likelihood import Likelihood
from model.ParentClasses.class_Priors_or_Posteriors_Handler import Priors_or_Posteriors_Handler
from model.Priors.Prior_selection import prior_x_selection, prior_y_selection
from model.PDEs.PDE_selection import PDE_selection

# jump prior / posterior
from model.Priors.class_PriorHyperparameter_Gamma import PriorHyperparameterGamma
from model.Priors.class_PriorX_Jumps import JumpPrior
from model.Priors.class_PriorX_Jumps_fixed import JumpPrior_fixed
from model.Posteriors.class_GammaPosterior_Hyperparameter import PosteriorHyperparameterGamma

# likelihood
from model.Likelihoods.class_VirtualLikelihood import VirtualLikelihood
from model.Likelihoods.class_ActualLikelihood import ActualLikelihood

# others
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
# import pyro
import os
import torch
import psutil
import warnings

# import all input parameters (this is, for some reason, only allowed on the modular level ...)
from model.input_preparation import *

from utils.function_part_observation import PartObservation
from model.function_noise import add_noise_dB
# from model.PDEs.PDE_selection import PDE_selection
from model.ParentClasses.class_nParametersToFields import nParametersToFields
from model.ParameterToField.ParameterToField_selection import ParameterToField_selection
from model.BCMasks.BCMask_selection import BCMask_selection
from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
import torch

from model.ParameterToField.class_AnalyticalLinearBasisFunctionTriangle import AnalyticalLinearBasisFunction
from model.ParameterToField.class_AnalyticalConstantBasisFunctionTriangle import AnalyticalConstantBasisFunction

# torch.set_default_dtype(torch.float64)

dir_name = None
opts_parameters = None

# Enable Latex in Plots
plt.rcParams['text.usetex'] = True
# clear parameter store from pyro_funcs
# pyro.clear_param_store()
# create a result directory
result_path = create_result_dict(dir_name=dir_name)
# save input file
copy_input_file("input.py", result_path)

### Device Selection ###
os.environ["CUDA_VISIBLE_DEVICES"] = str(cudaIndex)


if device == 'cpu':
    use_cuda = False
else:
    use_cuda = True

device = torch.device("cuda:0" if use_cuda else "cpu")
default_dtype = torch.float64
torch.set_default_device(device)
torch.set_default_dtype(default_dtype)


# %% Phantom specific
if load_data:
    u_obs_true = torch.load(load_data_path).to(device)
else:
    raise ValueError("No data loaded. This code has no implement way of generating data. Please provide ground truth data of the displacements.")

u_obs_true = u_obs_true[np.ix_(np.arange(0, 2, 1).astype(int), np.linspace(0,64,33), np.linspace(0,64,33))]
u_obs_true = u_obs_true.permute(0,2,1)

u_obs_true_1 = u_obs_true[0]
u_obs_true_2 = u_obs_true[1]
u_obs_true = torch.stack((u_obs_true_2, u_obs_true_1), dim=0)

y_true = u_obs_true.flatten()

# domain size
size_scale_x = L_x
size_scale_y = L_y
s_grid = regular_2D_mesh(101, 101, on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)

# deal with my programm stuff
observer = PartObservation()
observer.get_regular_index(0, 0,0, full_obs=True)

# obs positions
obs_positions = regular_2D_mesh(n_obs_s1, n_obs_s2, on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)
obs_positions2 = obs_positions.flatten(start_dim=-2, end_dim=-1).T

# noise
u_hat_filtered, sigma_u_hat = add_noise_dB(u_obs_true, SNR_in_dB, return_sigma=True)

plot_multiple_fields(u_obs_true, None, titles="data_obs", results_path=result_path)


# %%
# Create model
u_value = 0.1
if BC_given_explicitly:
    # bottom is fixed 
    my_given_values_mask_y1 = torch.zeros(Y1ToField_options["n_bfs_s1"], Y1ToField_options["n_bfs_s2"], dtype=torch.bool)
    my_given_values_mask_y2 = torch.zeros(Y2ToField_options["n_bfs_s1"], Y2ToField_options["n_bfs_s2"], dtype=torch.bool)
    my_given_values_mask_y1[:, 0] = True
    my_given_values_mask_y2[:, 0] = True
    BC1_given = BC2_given = torch.zeros(sum(my_given_values_mask_y1.flatten()), dtype=default_dtype)

    plot_multiple_fields(torch.stack((my_given_values_mask_y1, my_given_values_mask_y2)), None, titles=["BC1", "BC2"], results_path=result_path)

    if BC_given_y_flag:
        Y1ToField_options["mask_given_values"] = my_given_values_mask_y1
        Y1ToField_options["given_values"] = BC1_given
        Y1ToField_options["flag_given_values"] = True

        Y2ToField_options["mask_given_values"] = my_given_values_mask_y2
        Y2ToField_options["given_values"] = BC2_given
        Y2ToField_options["flag_given_values"] = True

        YToField_calc_options = Y1ToField_options.copy()
        YToField_calc_options["mask_given_values"] = torch.concat((Y1ToField_options["mask_given_values"], Y2ToField_options["mask_given_values"]))
        YToField_calc_options["given_values"] = torch.concat((Y1ToField_options["given_values"], Y2ToField_options["given_values"]))
    else: 
        YToField_calc_options = Y1ToField_options.copy()
        YToField_calc_options["flag_given_values"] = False

    if BC_given_theta_flag:
        Theta1ToField_options["mask_given_values"] = my_given_values_mask_y1
        Theta1ToField_options["given_values"] = torch.zeros_like(BC1_given)
        Theta1ToField_options["flag_given_values"] = True

        Theta2ToField_options["mask_given_values"] = my_given_values_mask_y2
        Theta2ToField_options["given_values"] = torch.zeros_like(BC2_given)
        Theta2ToField_options["flag_given_values"] = True

        ThetaToField_calc_options = Theta1ToField_options.copy()
        ThetaToField_calc_options["mask_given_values"] = torch.concat((Theta1ToField_options["mask_given_values"], Theta2ToField_options["mask_given_values"]))
        ThetaToField_calc_options["given_values"] = torch.concat((Theta1ToField_options["given_values"], Theta2ToField_options["given_values"]))
    else:
        ThetaToField_calc_options = Theta1ToField_options.copy()
        ThetaToField_calc_options["flag_given_values"] = False

N_el = 2 * (Y1ToField_options["n_bfs_s1"] - 1) * (Y1ToField_options["n_bfs_s2"] - 1)
# Select Parameter to Field maps
s_grid_options = {"nodes_s1": nodes_s1,
                    "nodes_s2": nodes_s2}
observation_options = {"n_obs_s1": n_obs_s1,
                        "n_obs_s2": n_obs_s2,
                        "SNR_in_dB": SNR_in_dB,
                        "from_Fenics": from_Fenics}
BC_Mask_options = {"x1": BC_mask_x1_options,
                   "x2": BC_mask_x2_options,
                   "y1": BC_mask_y1_options,
                   "y2": BC_mask_y2_options,
                   "theta1": BC_mask_theta1_options,
                   "theta2": BC_mask_theta2_options,
                   "xGT1": BC_mask_xGT1_options}
BC_Mask_kind = {"x1": BC_mask_x1_kind,
                "x2": BC_mask_x2_kind,
                "y1": BC_mask_y1_kind,
                "y2": BC_mask_y2_kind,
                "theta1": BC_mask_theta1_kind,
                "theta2": BC_mask_theta2_kind,
                "xGT1": BC_mask_xGT1_kind}
ParameterToField_kind = {"x1": X1ToField_kind,
                        "x2": X2ToField_kind,
                        "y1": Y1ToField_kind,
                        "y2": Y2ToField_kind,
                        "theta1": Theta1ToField_kind,
                        "theta2": Theta2ToField_kind,
                        "xGT1": XGT1ToField_kind}
ParameterToField_options = {"x1": X1ToField_options,
                            "x2": X2ToField_options,
                            "y1": Y1ToField_options,
                            "y2": Y2ToField_options,
                            "theta1": Theta1ToField_options,
                            "theta2": Theta2ToField_options,
                            "xGT1": XGT1ToField_options}

# %%
# Create model
print("Creating basis function grids ...")
# grid for my basis functions
bfs_grid_x1 = regular_2D_mesh(
    ParameterToField_options["x1"]["n_bfs_s1"], ParameterToField_options["x1"]["n_bfs_s2"], on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)
bfs_grid_x2 = None  # I have a constant field
bfs_grid_y1 = regular_2D_mesh(
    ParameterToField_options["y1"]["n_bfs_s1"], ParameterToField_options["y1"]["n_bfs_s2"], on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)
bfs_grid_y2 = regular_2D_mesh(
    ParameterToField_options["y2"]["n_bfs_s1"], ParameterToField_options["y2"]["n_bfs_s2"], on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)
bfs_grid_theta1 = regular_2D_mesh(
    ParameterToField_options["theta1"]["n_bfs_s1"], ParameterToField_options["theta1"]["n_bfs_s2"], on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)
bfs_grid_theta2 = regular_2D_mesh(
    ParameterToField_options["theta2"]["n_bfs_s1"], ParameterToField_options["theta2"]["n_bfs_s2"], on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)
bfs_grid_xGT1 = regular_2D_mesh(
    ParameterToField_options["xGT1"]["n_bfs_s1"], ParameterToField_options["xGT1"]["n_bfs_s2"], on_boundary=True, scale_x=size_scale_x, scale_y=size_scale_y)

print("Creating boundary condition masks ...")
def add_sgrid_to_options(BC_mask_options, s_grid):
    BC_mask_options["s_grid"] = s_grid
    return BC_mask_options

# prepare options dict for creating the boundary condition masks
BC_Mask_options["x1"] = add_sgrid_to_options(BC_Mask_options["x1"], s_grid)
BC_Mask_options["x2"] = add_sgrid_to_options(BC_Mask_options["x2"], s_grid)
BC_Mask_options["y1"] = add_sgrid_to_options(BC_Mask_options["y1"], s_grid)
BC_Mask_options["y2"] = add_sgrid_to_options(BC_Mask_options["y2"], s_grid)
BC_Mask_options["theta1"] = add_sgrid_to_options(BC_Mask_options["theta1"], s_grid)
BC_Mask_options["theta2"] = add_sgrid_to_options(BC_Mask_options["theta2"], s_grid)
BC_Mask_options["xGT1"] = add_sgrid_to_options(BC_Mask_options["xGT1"], s_grid)

# Known dirichlet boundary conditions with masks
BC_mask_x1 = BCMask_selection(BC_Mask_kind["x1"], BC_Mask_options["x1"])
BC_mask_x2 = BCMask_selection(BC_Mask_kind["x2"], BC_Mask_options["x2"])
BC_mask_y1 = BCMask_selection(BC_Mask_kind["y1"], BC_Mask_options["y1"])
BC_mask_y2 = BCMask_selection(BC_Mask_kind["y2"], BC_Mask_options["y2"])
BC_mask_theta1 = BCMask_selection(BC_Mask_kind["theta1"], BC_Mask_options["theta1"])
BC_mask_theta2 = BCMask_selection(BC_Mask_kind["theta2"], BC_Mask_options["theta2"])


print("Creating parameter to field maps ...")
# Prepare options dict for creating the fields
def field_options(field_args, s_grid, bfs_grid, BC_mask):
    options = {}
    options["bf_args"] = field_args
    options["s_grid"] = s_grid
    options["bf_grid"] = bfs_grid
    options["BC_mask"] = BC_mask
    return options

ParameterToField_options["x1"] = field_options(ParameterToField_options["x1"], s_grid, bfs_grid_x1, BC_mask_x1)
ParameterToField_options["x2"] = field_options(ParameterToField_options["x2"], s_grid, bfs_grid_x2, BC_mask_x2)
ParameterToField_options["y1"] = field_options(ParameterToField_options["y1"], s_grid, bfs_grid_y1, BC_mask_y1)
ParameterToField_options["y2"] = field_options(ParameterToField_options["y2"], s_grid, bfs_grid_y2, BC_mask_y2)
ParameterToField_options["theta1"] = field_options(ParameterToField_options["theta1"], s_grid, bfs_grid_theta1, BC_mask_theta1)
ParameterToField_options["theta2"] = field_options(ParameterToField_options["theta2"], s_grid, bfs_grid_theta2, BC_mask_theta2)
ParameterToField_options["xGT1"] = field_options(ParameterToField_options["xGT1"], s_grid, bfs_grid_xGT1, BC_mask_x1)
YToField_calc_options = field_options(YToField_calc_options, s_grid, bfs_grid_y1, BC_mask_y1)
ThetaToField_calc_options = field_options(ThetaToField_calc_options, s_grid, bfs_grid_theta1, BC_mask_theta1)

# Selection of the field typs + intialization
print("1. XToField ...")
X1ToField = ParameterToField_selection(ParameterToField_kind["x1"], ParameterToField_options["x1"])
X2ToField = ParameterToField_selection(ParameterToField_kind["x2"], ParameterToField_options["x2"])
XToField = nParametersToFields([X1ToField, X2ToField])#, [X1ToField.n_bfs])
print("2. YToField ...")
Y1ToField = ParameterToField_selection(ParameterToField_kind["y1"], ParameterToField_options["y1"])
Y2ToField = ParameterToField_selection(ParameterToField_kind["y2"], ParameterToField_options["y2"])
YToField = nParametersToFields([Y1ToField, Y2ToField]) #, [Y1ToField.n_bfs])
print("3. ThetaToField ...")
Theta1ToField = ParameterToField_selection(ParameterToField_kind["theta1"], ParameterToField_options["theta1"])
Theta2ToField = ParameterToField_selection(ParameterToField_kind["theta2"], ParameterToField_options["theta2"])
ThetaToField = nParametersToFields([Theta1ToField, Theta2ToField]) #, [Theta1ToField.n_bfs])
print("4. XGT1ToField ...")
XGT1ToField = ParameterToField_selection(ParameterToField_kind["xGT1"], ParameterToField_options["xGT1"])
XGTToField = nParametersToFields([XGT1ToField, X2ToField])

individual_fields = {"x1": X1ToField, 
                    "x2": X2ToField, 
                    "y1": Y1ToField, 
                    "y2": Y2ToField, 
                    "theta1": Theta1ToField, 
                    "theta2": Theta2ToField,
                    "xGT1": XGT1ToField}
# Selection of the field typs + intialization
print("1. XToField ...")
XToField_calc = AnalyticalConstantBasisFunction(ParameterToField_options["x1"])
print("2. YToField ...")
YToField_calc = AnalyticalLinearBasisFunction(YToField_calc_options)
print("3. ThetaToField ...")
ThetaToField_calc = AnalyticalLinearBasisFunction(ThetaToField_calc_options)

# My PDE
print("Creating PDE ...")
# make my input arrays to torch (cuda) tensors
for key, value in args_rhs.items():
    args_rhs[key] = torch.tensor(value, device=device)
args_rhs["rho"] = torch.tensor(rho, device=device)
args_rhs["frequency"] = torch.tensor(np.array([frequency], dtype=np.float32), device=device)

PDE_options =  {"nu": X2ToField_options["value"],
                "YToField": YToField_calc,
                "ThetaToField": ThetaToField_calc,
                "XToField": XToField_calc,
                "XGTToField": XGTToField,
                "args_rhs": args_rhs,
                "flag_E_plus_one": flag_E_plus_one}
my_PDE = PDE_selection(MaterialModel, PDE_options)


y_true = y_true[~YToField_calc_options["bf_args"]["mask_given_values"].flatten()]

# %%
# Create true values and observed values
# Ground truth
print("Creating ground truth ...")
from utils.torch_funcs.function_circular_inclusion import function_circular_inclusion
# Make a list of true x parameters
x_true = function_circular_inclusion(torch.ones_like(bfs_grid_x1)*2.3025, bfs_grid_x1, groundtruth_options)

u_true_fenics = YToField.eval(y_true).squeeze() # .transpose(dim0=-2, dim1=-1)  # flip the field
u_hat_full = u_true_fenics
u_true = u_true_fenics
MF_true = XGTToField.eval(x_true) # this should work for coefficients and fields

# the first part is a shortcut when I directly observe node values :)
if my_dim_y * my_dim_y == n_obs_s1 * n_obs_s2: 
    YToField_calc.flag_obs_nodes = True
else:
    YToField_calc._location_precompute(obs_positions2)
# %%
# Create stochastic objects
print("Creating priors ...")

# FIXME: Care here, I changed it so I have one MF as a constant
if 'prior_x_options' in locals():
    prior_x_options["dim"] = int(XToField_calc.num_unknowns)
    prior_x_options["dim_s_grid_1"] = s_grid.size()[1]
    prior_x_options["dim_s_grid_2"] = s_grid.size()[2]
    prior_x_options["dim_bf_grid1"] = ParameterToField_options["x1"]["bf_args"]["n_bfs_s1"]
    prior_x_options["dim_bf_grid2"] = ParameterToField_options["x1"]["bf_args"]["n_bfs_s2"]
    prior_x_options["dim_s_grid"] = s_grid.size()[1] * s_grid.size()[2]
    prior_x_options["X1ToField"] = None

posterior_x_options["dim_x"] = int(XToField_calc.num_unknowns)
prior_y_options["dim"] = posterior_y_options["dim_y"] = int(YToField_calc.num_unknowns)
# if ParameterToField_kind["x1"] == "ConstantTriangle":
#     prior_x_options["dim"] = XToField.len_parameter_list

# create priors for x and y
if 'prior_x_options' in locals():
    prior_x = prior_x_selection(prior_x_kind, prior_x_options)
prior_y = prior_y_selection(prior_y_kind, prior_y_options)

if flag_jump_prior:
    from utils.torch_funcs.function_regular_mesh import regular_2D_mesh
    bfs_grid_x1 = regular_2D_mesh(
        X1ToField_options["n_bfs_s1"], X1ToField_options["n_bfs_s2"], on_boundary=True)
    prior_jumps_options["XToField"] = XToField_calc
    if flag_penalty_learned:
        prior_jumps = JumpPrior(prior_jumps_options)
    else:
        prior_jumps = JumpPrior_fixed(prior_jumps_options)
    prior_jumps.set_neighbour_mask(YToField_calc.neighbours_mask)
    if flag_penalty_learned:
        prior_jumpPenalty = PriorHyperparameterGamma(prior_jumpPenalty_options)

if flag_noise_learned:
    from model.Priors.class_PriorNoise_Gamma import PriorNoiseGamma
    prior_noise = PriorNoiseGamma(prior_noise_options)
    if flag_learned_tau2:
        prior_noise_2 = PriorNoiseGamma(prior_noise_options)

priors = Priors_or_Posteriors_Handler()
# priors.add("x", prior_x, SpecialInput=prior_x_options["SpecialInput"])
priors.add("y", prior_y, SpecialInput=prior_y_options["SpecialInput"])
if flag_noise_learned:
    priors.add("tau", prior_noise, SpecialInput=prior_noise_options["SpecialInput"])
    if flag_learned_tau2:
        priors.add("tau_2", prior_noise_2, SpecialInput=None)

print("Creating likelihood ...")
# prepare options
actual_likelihood_options = {"YToField": YToField_calc, "sigma_u": sigma_u_hat}
SpecialInputs_actual_likelihood = ["y"]
if flag_noise_learned:
    SpecialInputs_actual_likelihood.append("tau")
if flag_learned_tau2 and flag_noise_learned:
    SpecialInputs_actual_likelihood2 = ["y", "tau_2"]
    actual_likelihood_options2 = actual_likelihood_options.copy()
    actual_likelihood_options2["which_half"] = "second"
    actual_likelihood_options["which_half"] = "first"

virtual_likelihood_options["dim_theta"] = ThetaToField_calc.num_unknowns
virtual_likelihood_options["PDE"] = my_PDE
virtual_likelihood_options["ThetaToField"] = ThetaToField_calc
# create likelihoods
likelihoods = Priors_or_Posteriors_Handler()
VirtLike = VirtualLikelihood(obsevations=torch.tensor(0.0), options=virtual_likelihood_options)
if not flag_learned_tau2:
    ActLike = ActualLikelihood(observations=u_hat_filtered, options=actual_likelihood_options)
    likelihoods.add("ActualLikelihood", ActLike, SpecialInput=SpecialInputs_actual_likelihood)
else:
    ActLike = ActualLikelihood(observations=u_hat_filtered[0], options=actual_likelihood_options)
    ActLike2 = ActualLikelihood(observations=u_hat_filtered[1], options=actual_likelihood_options2)
    likelihoods.add("ActualLikelihood", ActLike, SpecialInput=SpecialInputs_actual_likelihood)
    likelihoods.add("ActualLikelihood2", ActLike2, SpecialInput=SpecialInputs_actual_likelihood2)

print("Creating posteriors ...")
if "CNN" in posterior_x_kind:
    posterior_x_options["YToField"] = YToField
elif "FFN" in posterior_x_kind:
    posterior_x_options["dim_y"] = posterior_y_options["dim_y"]

# create my posterior for x&y and for lambda
posterior_X = posterior_selection_X(posterior_x_kind, posterior_x_options)
posterior_Y = posterior_selection_Y(posterior_y_kind, posterior_y_options)
if flag_noise_learned and flag_learned_tau2:
    posterior_noise_options["which_half"] = "first"
    posterior_noise_options2 = posterior_noise_options.copy()
    posterior_noise_options2["which_half"] = "second"
if flag_noise_learned:
    from model.Posteriors.class_GammaPosterior_noise import PosteriorNoiseGamma
    posterior_noise = PosteriorNoiseGamma(posterior_noise_options, prior_noise_options)
    if flag_learned_tau2:
        posterior_noise_2 = PosteriorNoiseGamma(posterior_noise_options2, prior_noise_options)
if flag_jump_prior and flag_penalty_learned:
    posterior_jumpPenalty = PosteriorHyperparameterGamma(posterior_jumpPenalty_options, prior_jumpPenalty_options)

posterior = Priors_or_Posteriors_Handler()
# posterior.add("x", posterior_X, SpecialInput=posterior_x_options["SpecialInput"])
posterior.add("y", posterior_Y, SpecialInput=posterior_y_options["SpecialInput"])
if flag_noise_learned:
    posterior.add("tau", posterior_noise, SpecialInput=posterior_noise_options["SpecialInput"])
    if flag_learned_tau2:
        posterior.add("tau_2", posterior_noise_2, SpecialInput=posterior_noise_options["SpecialInput"])

# load parameters from previous run if desired
if load_parameters:
    print("Loading parameters from {}".format(load_parameters_path))
    loaded_parameters = torch.load(load_parameters_path)
    posterior.load_state_dict(loaded_parameters, dont_load=dont_load)
    theta = loaded_parameters["theta"]
    if "u_hat_filtered" in loaded_parameters:
        u_hat_filtered = loaded_parameters["u_hat_filtered"]
    else:
        warnings.warn("u_hat_filtered not found in loaded parameters")

#%% Plots of the problem
print("Creating plots ...")
from utils.plots.plot_line_with_std_plus_sample import plot_line_with_std_plus_sample
plot_line_with_std_plus_sample(YToField, posterior_Y, true_field=u_true_fenics)

# plot
plot_multiple_fields(MF_true, s_grid, titles=[
                     r'$log(E)_{true}$', r'$nu_{true}$'], results_path=result_path)

# plot true values
plot_multiple_fields(u_true, s_grid,
                     titles=[r'$u^{true, torch}_1$', r'$u^{true, torch}_2$'],
                     results_path=result_path)

# %%
# Step 1: Learn N(y) (with fixed tau)

# Create optmizers
updated_y = False
update_tau = False

if flag_skip_opt_y:
    print("######################################################")
    warnings.warn("We are skipping the (seperate) Y optimization!")
    print("######################################################")
else:
    for counter_lr,lr_Y in enumerate(lrs_y):
        min_iter_y = min_iter_y_list[counter_lr]
        print("Creating optimizers ...")

        my_OptY = SVI(prior=priors,
                    likelihood=likelihoods,
                    posterior=posterior,
                    num_iter_Phi=num_iter_Y,
                    num_samples_Phi=num_sample_Y,
                    lr_Phi=lr_Y)

        print("Creating convergence criterions ...")
        # convergence criterions
        convergence_criterion_phi = convergence_criterion(
            when_check_convergence, tol_grad_ELBO_convergence)
        convergence_criterion_x_mean = convergence_criterion(
            when_check_convergence, tol_x_mean_convergence)
        convergence_criterion_y_mean = convergence_criterion(
            when_check_convergence, tol_y_mean_convergence)

        # running average for convergence criterion
        RA_calculator_phi = WelfordsOnlineAlgorithm()
        RA_calculator_theta = WelfordsOnlineAlgorithm()
        RA_calculator_x_mean = WelfordsOnlineAlgorithm()
        RA_calculator_y_mean = WelfordsOnlineAlgorithm()

        # create a reference set of weight functions to evaluate the actual residual
        theta_ref = torch.rand((10, ThetaToField_calc.num_unknowns))

        # create a new directory for the current temperature
        if update_tau:
            result_path_temp = os.path.join(
                result_path, "OPTY_{}_{:.2E}_tau".format(0.0, lr_Y))
        else:
            result_path_temp = os.path.join(
                result_path, "OPTY_{}_{:.2E}".format(0.0, lr_Y))
        os.makedirs(result_path_temp, exist_ok=True)

        for i in tqdm(range(max_MinMax_iterations)):
            # optimize Y (== parameters of the approximate posterior)
            # i.e. find the best parameter set that minimizes the ELBO while ignoring the residuals
            _elbo, _grad_elbo_phi_parts, elbo_parts, samples, intermediates = my_OptY.run()

            # logging
            if i == 0:
                my_temp_log = log()

            # logging
            if i % when_save_parameters == 0:
                # dummy theta, so my plots dont crash :)
                theta = torch.randn(ThetaToField_calc.num_unknowns)
                theta /= torch.linalg.norm(theta)

                my_temp_log.log(iteration=i,
                                # the GradStep method returns the grads in a list
                                grad_elbo_phi_parts=_grad_elbo_phi_parts,
                                grad_elbo_theta=torch.tensor([0.0, 0.0]), # not relevant
                                elbo_min=_elbo, # not relevant
                                elbo_max=_elbo, # not relevant
                                elbo_parts=elbo_parts,
                                theta=theta, # not relevant
                                Lambda=torch.tensor(0.0), # not relevant
                                Sqres=torch.tensor(0.0),  # not relevant
                                posterior_parameters=posterior.named_parameters,
                                JumpPenalty=torch.tensor(0.0))  # not relevant

            # update running average of log gradient of ELBO
            RA_phi = RA_calculator_phi.update(torch.log10(
                torch.abs(my_temp_log.grad_elbo_phi[-1])))
            # convergence_criterion_x_mean.update(_grad_elbo_phi_parts["x_0"])
            RA_y_mean = RA_calculator_y_mean.update(
                posterior.Posteriors["y"].y_0.clone().detach())

            # check for convergence
            if (i > 0) and (i % when_check_convergence == 0):
                # check for convergence for both theta and phi
                converged_phi, grad_grad_L_phi = convergence_criterion_phi.check_running_average_convergence(
                    RA_phi)
                converged_y_mean, grad_RA_y_mean = convergence_criterion_y_mean.check_running_average_convergence(
                    RA_y_mean)
                
                # print
                print("Iteration: {}". format(i))
                print("grad_grad_log_phi:   {} ({:.2E}) -- val: {:.2E}".format(
                    converged_phi, grad_grad_L_phi.item(), RA_phi.item()))
                if grad_RA_y_mean.isnan().any():
                    print("mean_y_0:            False (NAN)")
                else:
                    max_abs = '{:.3e}'.format(torch.max(torch.abs(grad_RA_y_mean)))
                    print('mean_y_0:            {} ({})'.format(
                        converged_y_mean, max_abs))
                print("Memory usage: {:.2f} GB".format(
                    psutil.Process(os.getpid()).memory_info().rss / 1024**3))

                # if both converged, break
                # & converged_theta:
                if converged_phi & converged_y_mean & (i > min_iter_y):
                    print("+++++++++++++++++++++++++++++++++++++++++++++++")
                    print("Converged!")
                    print("+++++++++++++++++++++++++++++++++++++++++++++++")
                    print()

                    # save torch parameters + all theta (relevant for Saddle point and Random residual) + current temperature
                    state_dict = posterior.state_dict()
                    state_dict["theta"] = theta # FIXME: For the greedy I want to save ALL thetas!
                    state_dict["Lambda"] = torch.tensor(0.0)
                    state_dict["u_hat_filtered"] = u_hat_filtered
                    torch.save(state_dict, result_path_temp + "/state_dict.pt")
                    my_temp_log.save(result_path_temp)

                    if observation_options["from_Fenics"]:
                        u_plot = u_true_fenics
                    else:
                        u_plot = u_true
                    postprocessing(my_temp_log,
                                    my_PDE,
                                    posterior,
                                    XToField,
                                    YToField,
                                    ThetaToField,
                                    s_grid,
                                    MF_true,
                                    u_plot,
                                    u_hat_full,
                                    sigma_u_hat,
                                    y_true,
                                    observer,
                                    result_path_temp)
                    updated_y = True

                    # after first iteration without updating tau -> start updating it
                    if flag_noise_learned and not update_tau:
                        update_tau = True
                        posterior.closedFormUpdateActive["tau"] = True
                        if flag_learned_tau2:
                            posterior.closedFormUpdateActive["tau_2"] = True
                        print("Starting to update tau ...")
                    # goes to the next temp_step
                    break


# %% Prep before calculation
posterior.add("x", posterior_X, SpecialInput=posterior_x_options["SpecialInput"])
if 'prior_x_options' in locals():
    priors.add("x", prior_x, SpecialInput=prior_x_options["SpecialInput"])
if flag_jump_prior:
    if flag_penalty_learned:
        priors.add("jumps", prior_jumps, SpecialInput=prior_jumps_options["SpecialInput"], active=True)
        priors.add("jumpPenalty", prior_jumpPenalty, SpecialInput=None, active=True)
        posterior.add("jumpPenalty", posterior_jumpPenalty, SpecialInput=None)
    else:
        priors.add("jumps", prior_jumps, SpecialInput=prior_jumps_options["SpecialInput"], active=True)
likelihoods.add("VirtualLikelihood", VirtLike, SpecialInput=virtual_likelihood_options["SpecialInput"])
# load parameters from previous run if desired
# 
if updated_y:
    if "y" not in dont_load:
        dont_load.append("y")
        print("######################################################")
        warnings.warn("Y was added to dont_load, as it was already trained!")
        print("######################################################")

if load_parameters: # then I can be sure, that I dont overload my learned y
    print("Loading parameters from {}".format(load_parameters_path))
    loaded_parameters = torch.load(load_parameters_path)
    posterior.load_state_dict(loaded_parameters, dont_load=dont_load)

# This is for a potential rescaling of the NN input of x
if ("FFN" in posterior_x_kind): # for FFNets
    # check if flag is existing and active
    if "rescale_input_flag" in posterior_x_options: 
        if posterior_x_options["rescale_input_flag"]:
            # check if q(y) is either calculated or loaded y
            if (updated_y) or (load_parameters and ("y" not in dont_load)): 
                print("Rescaling the input of the NN for x ...")
                samples = posterior_Y.sample(num_samples=100)
                mean_y = torch.mean(samples, dim=0)
                std_y = torch.std(samples, dim=0)
                posterior.Posteriors["x"].rescale_input_normal(mean_y, std_y) 
            else: 
                posterior.Posteriors["x"].rescale_input_flag = False

# %% OPT X + Y

my_OptPhi = SVI(prior=priors,
                likelihood=likelihoods,
                posterior=posterior,
                num_iter_Phi=num_iter_Phi,
                num_samples_Phi=num_sample_Phi,
                seperate_update=flag_phi_seperate_update,
                lr_Phi=lr_Phi,
                lr_X=lr_Phi_X,
                lr_Y=lr_Phi_X)

# posterior.Posteriors["y"].unfreeze_parameters()
print("Creating convergence criterions ...")
# convergence criterions
convergence_criterion_phi = convergence_criterion(
    when_check_convergence, tol_grad_ELBO_convergence)
convergence_criterion_x_mean = convergence_criterion(
    when_check_convergence, tol_x_mean_convergence)

# running average for convergence criterion
RA_calculator_phi = WelfordsOnlineAlgorithm()
RA_calculator_theta = WelfordsOnlineAlgorithm()
RA_calculator_x_mean = WelfordsOnlineAlgorithm()

# create a reference set of weight functions to evaluate the actual residual
theta_ref = VirtLike.get_random_theta(num_samples=50) #likelihood.get_random_theta(num_samples=50)

# %% Run combined optimization
update_tau = flag_learn_noise_first_iteration
update_theta = flag_learn_jump_first_iteration

posterior.closedFormUpdateActive["tau"] = update_tau
if flag_learned_tau2:
    posterior.closedFormUpdateActive["tau_2"] = update_tau
posterior.closedFormUpdateActive["theta"] = update_theta

for temp_step in range(num_temp_steps):
    # current temperature
    current_temp = VirtLike.lmbda

    if temp_step > 0:
        # set new temperature
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print("Iteration {:d} / {:d}: Old temperature: {:.2E}, New temperature: {:.2E}".format(
            temp_step + 1, num_temp_steps, current_temp, current_temp * temepering_factor))

        # posterior_lambda.set_value(current_temp * temepering_factor)
        current_temp = current_temp * temepering_factor
        VirtLike.lmbda = current_temp

        # set new learning rate for SVI
        if temp_step in lr_temp_steps:
            lr_old = my_OptPhi.lr
            my_OptPhi.lr = lr_old * lr_decrease_factor
            # since this criterion is lr dependent, we have to update it too
            if change_tol_w_step_size:
                convergence_criterion_x_mean.update_convergence_criterion(
                    when_check_convergence=when_check_convergence, tol=tol_x_mean_convergence * lr_decrease_factor)
            print(", Old learning rate: {:.2E}, New learning rate: {:.2E}".format(
                lr_old, my_OptPhi.lr))
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        # reset running averages
        RA_calculator_phi.reset()
        RA_calculator_theta.reset()

        # reset convergence criterions
        convergence_criterion_phi.reset()
        convergence_criterion_x_mean.reset()

        # reset ADAM optimizers
        my_OptPhi.reset()

    else:
        # set new temperature
        print("+++++++++++++++++++++++++++++++++++++++++++++++")
        print("Starting temperature: {:.2E}".format(current_temp))
        print("+++++++++++++++++++++++++++++++++++++++++++++++")

    # create a new directory for the current temperature
    folder = "temp_{}_{:.2E}".format(temp_step, current_temp)
    if update_tau:
        folder = folder + "_tau"
    if update_theta:
        folder = folder + "_theta"
    result_path_temp = os.path.join(
        result_path, folder)
    os.makedirs(result_path_temp, exist_ok=True)

    for i in tqdm(range(max_MinMax_iterations)):
        # optimize Phi (== parameters of the approximate posterior)
        # i.e. find the best parameter set that minimizes the ELBO while keeping theta constant
        _elbo, _grad_elbo_phi_parts, elbo_parts, samples, intermediates = my_OptPhi.run()

        # for consistent logging
        start_elbo = _elbo
        grad_elbo_phi_parts = _grad_elbo_phi_parts
        end_elbo = _elbo

        # logging
        if i == 0:
            my_temp_log = log()
            if temp_step == 0:
                my_global_log = log()

        # Calculate the ACTUAL residual for plotting
        # calculate the actual residual only every "when_check_convergence"-th iteration
        if i % when_check_convergence == 0:
            # 500 samples (x,y)  *  100 samples theta
            samples = posterior.sample(num_samples=50)
            res = my_PDE.forward(samples["x"], samples["y"], theta_ref)
            avrg_sq_res = torch.mean(torch.mean(
                torch.pow(res, 2), dim=0), dim=0).detach()

        # dummy theta, so my plots dont crash :)
        theta = torch.randn(ThetaToField_calc.num_unknowns)
        theta /= torch.linalg.norm(theta)

        # logging
        if i % when_save_parameters == 0:
            my_temp_log.log(iteration=i,
                            # the GradStep method returns the grads in a list
                            grad_elbo_phi_parts=grad_elbo_phi_parts,
                            grad_elbo_theta=torch.tensor([0.0, 0.0]),
                            elbo_min=end_elbo,
                            elbo_max=start_elbo,
                            elbo_parts=elbo_parts,
                            theta=theta,
                            Lambda=current_temp,
                            Sqres=avrg_sq_res,  # Sqres,
                            posterior_parameters=posterior.named_parameters,
                            JumpPenalty=torch.tensor(0.0))
            my_global_log.log(iteration=i,
                            # the GradStep method returns the grads in a list
                            grad_elbo_phi_parts=grad_elbo_phi_parts,
                            grad_elbo_theta=torch.tensor([0.0, 0.0]),
                            elbo_min=end_elbo,
                            elbo_max=start_elbo,
                            elbo_parts=elbo_parts,
                            theta=theta,
                            Lambda=current_temp,
                            Sqres=avrg_sq_res,  # Sqres,
                            posterior_parameters=posterior.named_parameters,
                            JumpPenalty=torch.tensor(0.0))
                            # save torch parameters + all theta (relevant for Saddle point and Random residual) + current temperature
            
            # create a new directory for the current iteration.
            result_path_temp_intermediate = os.path.join(result_path_temp, "iteration_{}".format(i))
            os.makedirs(result_path_temp_intermediate, exist_ok=True)
            state_dict = posterior.state_dict()
            state_dict["theta"] = theta # FIXME: For the greedy I want to save ALL thetas!
            state_dict["Lambda"] = current_temp
            state_dict["u_hat_filtered"] = u_hat_filtered
            state_dict["sigma_u_hat"] = sigma_u_hat
            if posterior_X.rescale_input_flag:
                state_dict["rescaled_input_mean_x"] = posterior_X.rescale_input_mean
                try:
                    state_dict["rescaled_input_std_x"] = posterior_X.rescale_input_std
                except:
                    pass
            # save first, so I still have the data, even when the postprocessing crashes
            torch.save(state_dict, result_path_temp_intermediate + "/state_dict_{}.pt".format(i))
            if observation_options["from_Fenics"]:
                u_plot = u_true_fenics
            else:
                u_plot = u_true
            postprocessing(my_temp_log,
                            my_PDE,
                            posterior,
                            XToField,
                            YToField,
                            ThetaToField,
                            s_grid,
                            MF_true,
                            u_plot,
                            u_hat_full,
                            sigma_u_hat,
                            y_true,
                            observer,
                            result_path_temp_intermediate)

        # update running average of log gradient of ELBO
        RA_theta = RA_calculator_theta.update(
            torch.log10(torch.abs(my_temp_log.grad_elbo_theta[-1])))
        RA_phi = RA_calculator_phi.update(torch.log10(
            torch.abs(my_temp_log.grad_elbo_phi[-1])))
        # convergence_criterion_x_mean.update(_grad_elbo_phi_parts["x_0"])
        if hasattr(posterior.Posteriors["x"], "x_0"):
            RA_x_mean = RA_calculator_x_mean.update(
                posterior.Posteriors["x"].x_0.clone().detach())

        # check for convergence
        if (i > 0) and (i % when_check_convergence == 0):
            # check for convergence for both theta and phi
            converged_phi, grad_grad_L_phi = convergence_criterion_phi.check_running_average_convergence(
                RA_phi)
            if hasattr(posterior.Posteriors["x"], "x_0"):
                converged_x_mean, grad_RA_x_mean = convergence_criterion_x_mean.check_running_average_convergence(
                    RA_x_mean)
            else:
                converged_x_mean = True
                grad_RA_x_mean = torch.tensor(0.0)
            # print
            print("Iteration: {}". format(i))
            print("grad_grad_log_phi:   {} ({:.2E}) -- val: {:.2E}".format(
                converged_phi, grad_grad_L_phi.item(), RA_phi.item()))
            if grad_RA_x_mean.isnan().any():
                print("mean_x_0:            False (NAN)")
            else:
                max_abs = '{:.3e}'.format(torch.max(torch.abs(grad_RA_x_mean)))
                print('mean_x_0:            {} ({})'.format(
                    converged_x_mean, max_abs))
            print("E[res^2]:            {:.2E}".format(avrg_sq_res))
            print("Memory usage: {:.2f} GB".format(
                psutil.Process(os.getpid()).memory_info().rss / 1024**3))
            if flag_noise_learned:
                print("tau:                 {:.2E}".format(
                    posterior.Posteriors["tau"].expectation()))
                if flag_learned_tau2:
                    print("tau_2:               {:.2E}".format(
                        posterior.Posteriors["tau_2"].expectation()))
            # make sure last temperature is fully converged out
            
            if temp_step == 0:
                min_iter = min_MinMax_iterations_first_iteration
            elif temp_step == num_temp_steps - 1:
                min_iter = min_MinMax_iterations_final_iteration
            else:
                min_iter = min_MinMax_iterations

            # if both converged, break
            # & converged_theta:
            if converged_phi & converged_x_mean & (i > min_iter):
                print("+++++++++++++++++++++++++++++++++++++++++++++++")
                print("Converged!")
                print("+++++++++++++++++++++++++++++++++++++++++++++++")
                print()

                # save torch parameters + all theta (relevant for Saddle point and Random residual) + current temperature
                state_dict = posterior.state_dict()
                state_dict["theta"] = theta # FIXME: For the greedy I want to save ALL thetas!
                state_dict["Lambda"] = current_temp
                state_dict["u_hat_filtered"] = u_hat_filtered
                state_dict["sigma_u_hat"] = sigma_u_hat
                if posterior_X.rescale_input_flag:
                    state_dict["rescaled_input_mean_x"] = posterior_X.rescale_input_mean
                    try:
                        state_dict["rescaled_input_std_x"] = posterior_X.rescale_input_std
                    except:
                        pass
                # save first, so I still have the data, even when the postprocessing crashes
                torch.save(state_dict, result_path_temp + "/state_dict.pt")
                my_temp_log.save(result_path_temp)
                if observation_options["from_Fenics"]:
                    u_plot = u_true_fenics
                else:
                    u_plot = u_true
                postprocessing(my_temp_log,
                               my_PDE,
                               posterior,
                               XToField,
                               YToField,
                               ThetaToField,
                               s_grid,
                               MF_true,
                               u_plot,
                               u_hat_full,
                               sigma_u_hat,
                               y_true,
                               observer,
                               result_path_temp)

                if flag_jump_prior:
                    update_theta = True
                    posterior.closedFormUpdateActive["jumpPenalty"] = True
                if flag_learn_noise_second_iteration and update_tau:
                    update_tau = True
                    posterior.closedFormUpdateActive["tau"] = True
                    if flag_learned_tau2:
                        posterior.closedFormUpdateActive["tau_2"] = True
                elif update_tau:
                    flag_learn_noise_second_iteration = True

                # goes to the next temp_step
                break


print("+++++++++++++++++++++++++++++++++++++++++++++++")
print("Reached maximum temperature!")
print("+++++++++++++++++++++++++++++++++++++++++++++++")
result_path_temp = os.path.join(result_path, "global")
os.mkdir(result_path_temp)

# %%
# Postprocessing
for name, value in my_global_log.elbo_parts.items():
    my_global_log.elbo_parts[name] = torch.stack(
        my_global_log.elbo_parts[name])
for name, value in my_global_log.grad_elbo_phi_parts.items():
    my_global_log.grad_elbo_phi_parts[name] = torch.stack(
        my_global_log.grad_elbo_phi_parts[name])
my_global_log.iteration = torch.arange(0, len(my_global_log.iteration)).cpu()

postprocessing(my_global_log,
               my_PDE,
               posterior,
               XToField,
               YToField,
               ThetaToField,
               s_grid,
               MF_true,
               u_plot,
               u_hat_full,
               sigma_u_hat,
               y_true,
               observer,
               result_path_temp)
