import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np

# my imports
from model.ParentClasses.class_approximatePosterior import approximatePosterior
from utils.torch_funcs.function_torch_split_combine import torch_split, torch_combine
from utils.torch_funcs.function_make_torch_tensor import make_torch_tensor


class DiagMVNPosterior_X(approximatePosterior):
    def __init__(self, options):
        # inherit from approximatePosterior
        super().__init__(options)
        
        # dim of the MVN
        self.dim_x = options["dim_x"]

        # means & coveriance
        self.x_0 = nn.Parameter(make_torch_tensor(options["mean_x_0"], torch.Size([self.dim_x])))
        self.cov_x_parameters = nn.Parameter(make_torch_tensor(torch.log(torch.tensor(options["cov_x_0"])), torch.Size([self.dim_x])))

    def sample(self, num_samples=None):
        if num_samples==None:
            x = dist.normal.Normal(self.x_0, self.calc_scale_tril()).rsample()
        else:
            x = dist.normal.Normal(self.x_0, self.calc_scale_tril()).rsample(torch.Size((num_samples,)))
        return x

    def log_prob(self, x):
        # calculate log_prob
        log_prob_x = dist.normal.Normal(self.x_0,self.calc_scale_tril()).log_prob(x)
        return log_prob_x
    
    def entropy(self):
        # calculate entropy
        entropy_x = dist.normal.Normal(self.x_0, self.calc_scale_tril()).entropy()
        entropy_x = torch.sum(entropy_x)
        return entropy_x

    @property
    def mean_x(self):
        return self.x_0

    @property
    def mean_y(self, x=None):
        return self.y_0

    def calc_scale_tril(self):
        # compute torche covariance matrix
        # diagonal elements are exponentiated
        L = torch.exp(self.cov_x_parameters)
        # return torche covariance matrix
        return L
        
