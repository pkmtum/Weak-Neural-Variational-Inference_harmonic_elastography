import torch
from model.ParentClasses.class_ParentPriors import PriorX

class PriorNoiseGamma(PriorX):
    def __init__(self, prior_hyperparameter_options):
        # inherit from PriorX
        super().__init__(prior_hyperparameter_options)
        # extract options
        self.a_0 = self.options["a_0"]
        self.b_0 = self.options["b_0"]

    def sample(self, num_samples):
        return torch.distributions.Gamma(self.a_0, self.b_0).sample(num_samples)
    
    def log_prob(self, tau=None, tau_2=None):
        if tau is None:
            tau = tau_2
        return torch.distributions.Gamma(self.a_0, self.b_0).log_prob(tau).sum(dim=-1)

