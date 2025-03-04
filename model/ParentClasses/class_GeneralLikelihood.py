import torch
import torch.nn as nn

class GeneralLikelihood(nn.Module):
    def __init__(self, observations, options):
        super().__init__()
        self.observations = observations
        self.options = options

    def log_prob(self):
        pass
    
    def set_observation(self, observations):
        self.observations = observations
    
    def get_observation(self):
        return self.observations