import torch

class OptLambda:
    def __init__(self,
                 posterior,
                 posterior_lambda,
                 PDE,
                 num_samples_lambda):

        # posteriors
        self.posterior = posterior
        self.posterior_lambda = posterior_lambda

        # PDE (to calc residual)
        self.PDE = PDE

        # number of samplesto calc E(r^2)
        self.num_samples_lambda = num_samples_lambda

    def run(self, theta):
        # get samples of x, y
        x, y = self.posterior.sample(num_samples = self.num_samples_lambda)

        # calculate residuals
        res = self.PDE.forward(x, y, theta)

        # Square them
        Sqres = torch.pow(res, 2)

        # squared residual
        E_r_max_squared = torch.mean(Sqres)

        # update the parameters
        self.posterior_lambda.update_parameters(E_r_max_squared)
