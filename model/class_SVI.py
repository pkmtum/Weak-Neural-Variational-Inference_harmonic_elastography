import torch
import torch.nn


class SVI:
    def __init__(self,
                 prior,
                 likelihood,
                 posterior,
                 num_iter_Phi,
                 num_samples_Phi,
                 seperate_update=False,
                 lr_Phi=None,
                 lr_X=None,
                 lr_Y=None):
        # these are my Priors (from prior classes)
        self.prior = prior

        # my likelihood
        # self.likelihood = likeihood

        # my posterior (from posterior class)
        self.posterior = posterior

        # likelihoods
        self.likelihoods = likelihood

        # Optimization parameters
        # Number of SVI iterations
        self.num_iter_Phi = num_iter_Phi
        self.num_samples_Phi = num_samples_Phi
        
        # Opimtizer
        self.seperate_update = seperate_update
        if not self.seperate_update:
            self.lr = lr_Phi
            self.opimizer = torch.optim.Adam(
                self.posterior.parameters, self.lr, maximize=True) 
        else:
            self.lr_X = lr_X
            self.lr_Y = lr_Y
            self.opimizer_X = torch.optim.Adam(
                self.posterior.Posteriors["x"].parameters(), self.lr_X, maximize=True)
            self.opimizer_Y = torch.optim.Adam(
                self.posterior.Posteriors["y"].parameters(), self.lr_Y, maximize=True)

    def reset(self):
        if not self.seperate_update:
            self.opimizer = torch.optim.Adam(
                self.posterior.parameters, self.lr, maximize=True)
        else:
            self.opimizer_X = torch.optim.Adam(
                self.posterior.Posteriors["x"].parameters(), self.lr_X, maximize=True)
            self.opimizer_Y = torch.optim.Adam(
                self.posterior.Posteriors["y"].parameters(), self.lr_Y, maximize=True)

    def ELBO(self):
        # -------------------------------------------------------------
        # sample from the posterior
        samples = self.posterior.sample(num_samples=self.num_samples_Phi)

        # -------------------------------------------------------------
        # calculate the expected log likelihood
        E_log_L, _ = self.likelihoods.expected_log_prob(samples)
        summed_E_log_L = sum(list(E_log_L.values()))

        # log prob of priors
        E_log_prior, _ = self.prior.expected_log_prob(samples)
        summed_E_log_prior = sum(list(E_log_prior.values()))

        # entropy (of q)
        E_entropy, _ = self.posterior.expected_entropy(samples)
        summed_E_entropy = sum(list(E_entropy.values()))

        # ELBO as the sum of all log probabilities
        ELBO = summed_E_log_L + summed_E_log_prior + summed_E_entropy

        # -------------------------------------------------------------
        # Closed form updates
        # collect intermediates (residuals, jumps, etc.)
        intermediates_q = self.posterior.collect_intermediates()
        intermediates_L = self.likelihoods.collect_intermediates()
        intermediates_P = self.prior.collect_intermediates()
        intermediates = {**intermediates_q, **intermediates_L, **intermediates_P}

        # update the posterior in closed form
        self.posterior.closed_form_update(samples, intermediates)

        # -------------------------------------------------------------
        # ELBO parts + return
        # Likelihoods
        elbo_parts = dict()
        for name, E_log_L_i in E_log_L.items():
            elbo_parts["E_log_L_{}".format(
                name)] = E_log_L_i.clone().detach()
        # Priors
        for name, E_log_prior_i in E_log_prior.items():
            elbo_parts["E_log_prior_{}".format(
                name)] = E_log_prior_i.clone().detach()
        # Entropies
        for name, E_entropy_i in E_entropy.items():
            elbo_parts["E_entropy_{}".format(
                name)] = E_entropy_i.clone().detach()

        for name, sample in samples.items():
            samples[name] = sample.clone().detach()

        return ELBO, elbo_parts, samples, intermediates

    def run(self):
        # reset the optimizer
        if not self.seperate_update:
            self.opimizer.zero_grad()
        else:
            self.opimizer_X.zero_grad()
            self.opimizer_Y.zero_grad()

        # Do SVI
        elbo, elbo_parts, samples, intermediates = self.ELBO()

        # get the gradient
        elbo.backward(retain_graph=True)

        # do a step
        if not self.seperate_update:
            self.opimizer.step()
        else:
            self.opimizer_X.step()
            self.opimizer_Y.step()

        # get the current gradient
        current_grad = dict()
        for name, param in self.posterior.named_parameters:
            if param.requires_grad:  # there may be frozen parameters that do not give a gradient
                current_grad[name] = param.grad.clone().detach()

        return elbo.clone().detach(), current_grad, elbo_parts, samples, intermediates