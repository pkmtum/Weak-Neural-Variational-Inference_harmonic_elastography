import torch
import torch.nn as nn
import torch.distributions as dist

from model.ParentClasses.class_GeneralLikelihood import GeneralLikelihood
import math

class VirtualLikelihood(GeneralLikelihood):
    def __init__(self, obsevations, options) -> None:
        super().__init__(obsevations, options)

        # nessassary information (that should throw an error if not given)
        self.dim_theta = options["dim_theta"]
        self.PDE = options["PDE"]
        self.lmbda = options["lmbda"]
        self.sigma_r = torch.sqrt(torch.tensor(1/self.lmbda))
        self.num_samples_theta = options["num_samples_theta"]

        # choose one kind of weight function
        kind = options.get("samples_theta_kind", "Random")
        if kind == "Random":
            self.get_theta_custom = self.get_random_theta
        elif kind == "Full":
            self.get_theta_custom = self.get_full_theta
            self.full_theta = torch.eye(self.dim_theta)
        elif kind == "Circular":
            self.get_theta_custom = self.get_circular_theta
            self.min_rad = torch.tensor(options["min_rad"])
            self.max_rad = torch.tensor(options["max_rad"])
            self.radius_type = options.get("radius_type", "linear")
            self.ThetaToField = options["ThetaToField"]
            if self.radius_type not in ["linear", "log"]:
                raise ValueError("Radius type must be 'linear' or 'log'")
        elif kind == "RandomFull":
            self.get_theta_custom = self.get_random_full_theta
            self.full_theta = torch.eye(self.dim_theta)
        else:
            raise ValueError("Kind must be 'Random', 'Full' or 'Circular'")
        
        # choose one kind of likelihood
        self.likelihood_kind = options["likelihood_type"]

        # others
        self.working_theta = nn.Parameter(torch.rand(self.dim_theta))
        self.theta = torch.tensor([])
        self.r = None

    def log_prob(self, x, y, lmbda=None):
        # Wikipedia: "Given no event (no data), the likelihood is 1; any non-trivial event will have a lower likelihood."
        # --> Thus log(1) = 0
        if self.num_samples_theta == 0:
            return torch.tensor(0.)
        
        # if we infer lambda (precision)
        if lmbda is not None:
            sigma_r = torch.sqrt(torch.tensor(1/lmbda))
        else:
            sigma_r = self.sigma_r

        # sample theta
        theta = self.get_theta_custom(num_samples=self.num_samples_theta)

        # residual calculation for mulitple (or one) theta
        res = self.PDE.forward(x, y, theta)
        
        #save intermediates
        self.r = res

        # calculate log_prob
        if self.likelihood_kind == "Gaussian":
            log_prob = dist.Normal(torch.tensor(0.0), sigma_r).log_prob(res).sum(dim=-1) # sum over all thetas
        elif self.likelihood_kind == "Laplace":
            log_prob = dist.Laplace(torch.tensor(0.0), sigma_r).log_prob(res).sum(dim=-1)
        elif self.likelihood_kind == "Alternative":
            raise NotImplementedError("Alternative likelihood not implemented yet.")
        else:
            raise ValueError("Likelihood type must be 'Gaussian', 'Laplace' or 'Alternative'")

        return log_prob

    def add_theta(self, theta):
        if theta.dim() == 1:
            theta = theta.unsqueeze(0)
        if self.num_theta == 0:
            self.theta = theta
        else:
            self.theta = torch.concat((self.theta, theta), axis=0)

    def get_theta(self):
        if self.theta.numel() == 0:
            return list(self.parameters())[0].clone().detach().unsqueeze(0)
        else:
            return self.theta
    
    def reset_theta(self):
        self.theta = torch.tensor([])

    def reset_working_theta(self, theta_0):
        self.working_theta = nn.Parameter(theta_0)

    def set_full_theta(self):
        self.theta = torch.eye(self.working_theta.clone().detach().size()[0])

    def get_random_theta(self, num_samples=1, mean=0, std=1):
        return torch.randn(num_samples, self.working_theta.clone().detach().size()[0]) * std + mean

    def get_random_full_theta(self, num_samples=1):
        num = torch.randint(0, self.dim_theta, (num_samples,))
        return self.full_theta[num]
    
    def get_full_theta(self):
        return self.full_theta

    def get_circular_theta(self, num_samples=1):
        # create a weight function with a random center and a radius
        bfs_per_row = self.ThetaToField.bf_grid.size()[1]
        bfs_per_col = self.ThetaToField.bf_grid.size()[2]
        bf_grid = self.ThetaToField.bf_grid
        n_bfs = self.ThetaToField.n_bfs
        BC_mask = getattr(self.ThetaToField, 'mask_given_values', None)
        # random center
        x, y = torch.randint(0, bfs_per_row, (num_samples,)), torch.randint(0, bfs_per_col, (num_samples,))
        center = bf_grid[:, x, y]
        # radius
        if self.radius_type == "linear":
            radius = torch.rand(num_samples) * (self.max_rad - self.min_rad) + self.min_rad
        elif self.radius_type == "log":
            radius = torch.exp(torch.rand(num_samples) * (torch.log(self.max_rad) - torch.log(self.min_rad)) + torch.log(self.min_rad))
        # calculate squared distance to center
        dist_center = torch.sub(bf_grid.repeat(num_samples,1,1,1).permute(2,3,1,0), center)
        sq_dist_center = torch.pow(dist_center, 2)
        d = sq_dist_center.sum(dim=2)
        # which points are in the circle
        mask = d <= radius ** 2
        points = mask.flatten(start_dim=0, end_dim=1)
        # flip the coin if this is a "x" or "y" - dimension theta
        flip = torch.randint(0, 2, (num_samples,)).to(torch.bool)
        # create theta
        theta = torch.zeros(n_bfs, num_samples, dtype=torch.bool)
        theta[n_bfs//2:, flip] = points[:, flip]
        theta[:n_bfs//2, ~flip] = points[:, ~flip]
        # delete elements where we have Dirichlet BCs
        if BC_mask is not None:
            theta = theta[~BC_mask, :]
        # to float and correct format
        theta = theta.to(torch.double)
        theta = theta.permute(1,0)
        return theta

    def collect_intermediates(self):
        return self.r, "r"

    @property
    def num_theta(self):
        return self.theta.size(dim=0)