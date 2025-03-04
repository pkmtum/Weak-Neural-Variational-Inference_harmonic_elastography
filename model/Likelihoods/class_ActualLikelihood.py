import torch
import torch.distributions as dist

from model.ParentClasses.class_GeneralLikelihood import GeneralLikelihood

class ActualLikelihood(GeneralLikelihood):
    def __init__(self, observations, options) -> None:
        super().__init__(observations, options)

        # nessassary information (that should throw an error if not given)
        self.YToField = options["YToField"]
        self.sigma_u = options["sigma_u"]
        if self.sigma_u == 0 or self.sigma_u is None:
            self.tau_u = None
        else:
            self.tau_u = 1/self.sigma_u**2
        self.flat_observations = self.observations.flatten()
        self.which_half = options.get("which_half", "both")

        # intermediates
        self.u_minus_uhat = None

    def log_prob(self, y, tau=None, tau_2=None):
        # calculate the data at the respective locations
        u = self.YToField.eval_at_locations(y)
        if self.which_half == "both":
            u_flat = u.flatten(start_dim=-2)
        elif self.which_half == "first":
            u_flat = u[:, 0]
        elif self.which_half == "second":
            u_flat = u[:, 1]

        # if we infer tau (precision)
        if tau is not None:
            sigma_u = torch.sqrt(torch.tensor(1/tau))
        elif tau_2 is not None:
            sigma_u = torch.sqrt(torch.tensor(1/tau_2))
        else:
            sigma_u = self.sigma_u

        # save intermediates
        self.u_minus_uhat = u_flat - self.flat_observations

        # calculate the log likelihood
        log_prob = dist.Normal(u_flat, sigma_u).log_prob(self.flat_observations).sum(dim=-1) # sum over the dimensions of the field

        return log_prob
    
    def collect_intermediates(self):
        if self.which_half == "both":
            return self.u_minus_uhat, "u_minus_uhat"
        elif self.which_half == "first":
            return self.u_minus_uhat, "u_minus_uhat_first"
        elif self.which_half == "second":
            return self.u_minus_uhat, "u_minus_uhat_second"
        else:
            raise ValueError("which_half should be 'both', 'first' or 'second'")
