import torch
import numpy as np
import math
from utils.torch_funcs.function_cuda_to_cpu import cuda_to_cpu
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import grad
# my imports
# from utils.function_power_iteration import power_iteration
# from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine


class OptTheta:
    def __init__(self, PDE, posterior, likelihood, num_iter_Theta: int, lr_Theta, num_sample_theta):
        # general problem setup
        self.PDE = PDE
        self.q = posterior
        # self.p_x = prior_x
        # self.p_y = prior_y
        self.L = likelihood
        # self.sigma_r = likelihood.sigma_r

        # optimization parameters
        self.num_iter_Theta = num_iter_Theta
        self.num_sample_theta = num_sample_theta
        self.lr_Theta = lr_Theta

        # initialize my model objective parameter for ADAM
        # theta_0 = theta_0.clone().detach()
        # theta_0 = theta_0 / torch.linalg.norm(theta_0)

        # samples (num_samples, x, y) tuple from q_phi --> only once for IS
        # x, y = self.q.sample(num_samples=self.num_sample_theta)

        # part of the importance sampling weights which is constant for all theta
        # log_IS_weights_constant = self.get_IS_weights_constant(x, y)  # in log scale

        # ONLY the unconstrained parameters should be trained i.e. requires grad == True; while theta_constrained
        # requires no grad
        # theta_0.requires_grad_(True)

        # ADAM
        self.model = list(self.L.parameters())
        self.optimizer = torch.optim.Adam(
            self.model, lr=self.lr_Theta, maximize=True)

    def reset(self):
        # ADAM
        self.optimizer = torch.optim.Adam(
            self.model, lr=self.lr_Theta, maximize=True)

    def theta_general_gpu(self):
        # Gradient optimization loop
        for i in range(self.num_iter_Theta):
            # reset gradients from previous iteration
            self.optimizer.zero_grad()

            # norm the unconstrained part of the weighting function to the length it had before
            theta_normed = self.model[0] / torch.linalg.norm(self.model[0])
            theta_normed = theta_normed.unsqueeze(0)

            # sample
            samples = self.q.sample(num_samples=self.num_sample_theta)
            x = samples["x"]
            y = samples["y"]
            # get ln( p(r_{N+1} | z, theta_{N+1}) ) = -1/(2 \sigma**2) * r_1000{N+1}^2
            # r_{N+1} for different z_i but only theta_{N+1}
            res = self.PDE.forward(x, y, theta_normed)

            # create squared residual
            Sqres = torch.pow(res, 2)

            # get MC estiamte
            # optimize wrt to the log of the MC estimate, since the MC estimate can be very small.
            # This way we can avoid underflow errors as well as the weighting function getting stuck
            loss = torch.log(torch.mean(Sqres))

            # using ADAM
            if i == int(self.num_iter_Theta - 1):
                # This exception is needed to get the gradient at the same point as the other optimizer
                loss.backward(retain_graph=True)
                # for the plot we need the gradient d MC_estimate / d theta NOT d log(MC_estimate) / d theta
                nabla_theta_ELBO = self.model[0].grad.clone(
                ).detach() * torch.exp(loss.clone().detach())
            else:
                loss.backward(retain_graph=False)
            self.optimizer.step()

            # norming theta
        theta = self.model[0] / torch.linalg.norm(self.model[0])

        # check if everything worked out with the norming
        assert torch.isclose(torch.tensor(1.0), torch.linalg.norm(theta))

        # return detached (so it won't break or slow down anything)
        return theta.clone().detach(), nabla_theta_ELBO, Sqres.clone().detach()
    
    # def theta_general_cpu(self, theta_0):
    #     # This function aims to find the theta (i.e. normed weighting function) that maximizes the expectation of the
    #     # squared residual, BUT given the boundary conditions
    #     # Dirichlet -> weighting function is zero
    #     # Neumann -> Term in residual

    #     # initialize my model objective parameter for ADAM
    #     theta_0 = theta_0.clone().detach()
    #     theta_0 = theta_0 / torch.linalg.norm(theta_0)

    #     # ONLY the unconstrained parameters should be trained i.e. requires grad == True; while theta_constrained
    #     # requires no grad
    #     theta_0.requires_grad_(True)

    #     # ADAM
    #     model = [theta_0]
    #     optimizer = torch.optim.Adam(model, lr=self.lr_Theta, maximize=True)

    #     # Gradient optimization loop
    #     for i in range(self.num_iter_Theta):
    #         # reset gradients from previous iteration
    #         optimizer.zero_grad()

    #         # norm the unconstrained part of the weighting function to the length it had before
    #         theta_normed = model[0] / torch.linalg.norm(model[0])

    #         # create Monte Carlo samples
    #         for j in range(self.num_sample_theta):
    #             # samples (x, y) tuple from q_phi
    #             x, y = self.q.sample()

    #             # calculate ONE residual using (x, y)
    #             # I think theta == weighting function for the fenics_funcs context!
    #             # usually: w_theta = sum theta * shape_func
    #             # where shape_fun is e.g. Lagrange of first order.
    #             # But I only provide the coefficients and FEniCS takes care of the rest - I hope :)
    #             # loss function == squared residual
    #             res = self.PDE.forward(x, y, theta_normed)

    #             if j == 0:
    #                 loss = torch.pow(res, 2) / self.num_sample_theta
    #             else:
    #                 loss += torch.pow(res, 2) / self.num_sample_theta

    #         # using ADAM
    #         if i == int(self.num_iter_Theta - 1):
    #             # This exception is needed to get the gradient at the same point as the other optimizer
    #             loss.backward(retain_graph=True)
    #             nabla_theta_ELBO = model[0].grad.clone().detach()
    #         else:
    #             loss.backward()
    #         optimizer.step()

    #     # norming theta
    #     theta = model[0] / torch.linalg.norm(model[0])

    #     # check if everything worked out with the norming
    #     assert torch.isclose(torch.tensor(1.0), torch.linalg.norm(theta))

    #     # return detached (so it won't break or slow down anything)
    #     return theta.clone().detach(), nabla_theta_ELBO

    # def theta_constrained(self, theta_0, idx_constrained):
    #     # This function aims to find the theta (i.e. normed weighting function) that maximizes the expectation of the
    #     # squared residual, BUT given the boundary conditions
    #     # Dirichlet -> weighting function is zero
    #     # Neumann -> Term in residual

    #     # initialize my model objective parameter for ADAM
    #     theta_0 = theta_0.clone().detach()
    #     theta_0 = theta_0 / torch.linalg.norm(theta_0)

    #     # now perform a split into parameters, which are constrained, and those which are not
    #     theta_unconstrained, theta_constrained = torch_split(theta_0, idx_constrained)
    #     my_torch_combine = torch_combine(len(theta_0), idx_constrained)

    #     # we later need to do the norm. To make sure only the constrained parameters get scaled, we need to scale
    #     # them to the length they have now
    #     theta_unconstrained_norm = torch.linalg.norm(theta_unconstrained)

    #     # ONLY the unconstrained parameters should be trained i.e. requires grad == True; while theta_constrained
    #     # requires no grad
    #     theta_unconstrained.requires_grad_(True)

    #     # ADAM
    #     model = [theta_unconstrained]
    #     optimizer = torch.optim.Adam(model, lr=self.lr_Theta, maximize=True)

    #     # Gradient optimization loop
    #     for i in range(self.num_iter_Theta):
    #         # reset gradients from previous iteration
    #         optimizer.zero_grad()

    #         # norm the unconstrained part of the weighting function to the length it had before
    #         theta_unconstrained_normed = model[0] / torch.linalg.norm(model[0]) * theta_unconstrained_norm

    #         # now we can recombine the constrained and unconstrained part (they are together normed to 1)
    #         theta = my_torch_combine.apply(theta_unconstrained_normed, theta_constrained)

    #         # create Monte Carlo samples
    #         for j in range(self.num_sample_theta):
    #             # samples (x, y) tuple from q_phi
    #             x, y = self.q.sample()

    #             # calculate ONE residual using (x, y)
    #             # I think theta == weighting function for the fenics_funcs context!
    #             # usually: w_theta = sum theta * shape_func
    #             # where shape_fun is e.g. Lagrange of first order.
    #             # But I only provide the coefficients and FEniCS takes care of the rest - I hope :)
    #             # loss function == squared residual
    #             res = self.PDE.forward(x, y, theta, theta_opt=True)

    #             if j == 0:
    #                 loss = torch.pow(res, 2)/self.num_sample_theta
    #             else:
    #                 loss += torch.pow(res, 2)/self.num_sample_theta

    #         # using ADAM
    #         if i == int(self.num_iter_Theta - 1):
    #             # This exception is needed to get the gradient at the same point as the other optimizer
    #             loss.backward(retain_graph=True)
    #             nabla_theta_ELBO = model[0].grad.clone().detach()
    #         else:
    #             loss.backward()
    #         optimizer.step()

    #     # norming theta
    #     theta_unconstrained_normed = model[0] / torch.linalg.norm(model[0]) * theta_unconstrained_norm
    #     theta = my_torch_combine.apply(theta_unconstrained_normed, theta_constrained)
    #     # check if everything worked out with the norming
    #     assert torch.isclose(torch.tensor(1.0), torch.linalg.norm(theta))

    #     # return detached (so it won't break or slow down anything)
    #     return theta.clone().detach(), nabla_theta_ELBO

    # def theta_closed_form(self, OptMode="unconstrained"):
    #     for i in range(self.num_sample_theta):
    #         # samples (x, y) tuple from q_phi
    #         x, y = self.q.sample()

    #         # true weighting function
    #         if i == 0:
    #             theta_true_sample = self.PDE.all_residuals(x, y, mode=OptMode)
    #         else:
    #             theta_true_sample += self.PDE.all_residuals(x, y, mode=OptMode)

    #     theta_true = theta_true_sample / self.num_sample_theta

    #     theta_true = theta_true / torch.linalg.norm(theta_true)

    #     return theta_true.clone().detach(), torch.zeros(theta_true.size())

    # def theta_power_iteration(self, num_samples):
    #     # This function aims to find the theta (i.e. weighting function) that maximizes the expectation of the
    #     # squared residual. This is a shortcut (I guess) using power iteration if the weighting function is a linear
    #     # combination of feature functions like: w_theta = theta.T u
    #     # BTW: I am pretty sure this is slower in my case, since it requires assembling the whole FE system.
    #
    #     # sample from my approximate posterior q
    #     q_samples = self.q.sample(num_samples)
    #
    #     # MC to get C = expectation of c (== all residuals squared)
    #     C = torch_funcs.zeros((self.PDE.V.dim(), self.PDE.V.dim()))
    #     for i in range(num_samples):
    #         # get all (?!) residuals depending on x, y sampled from q
    #         all_res = self.PDE.all_residuals(q_samples[i, 0], q_samples[i, 1])
    #
    #         # calculate semi-positive definite matrix c(x,y)
    #         c = torch_funcs.outer(all_res, all_res.T)
    #
    #         # sum up for MC
    #         C += c
    #
    #     # MC estimate
    #     C = C / num_samples
    #
    #     # Power iteration: finding the largest eigenvalue and associated eigenvector (normed to 1 already)
    #     E_SqRes, theta = power_iteration(C)
    #
    #     return E_SqRes, theta
