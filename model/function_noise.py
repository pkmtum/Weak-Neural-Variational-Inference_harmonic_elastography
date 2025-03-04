import torch
import numpy as np


def add_noise_dB(u_ref, SNR_in_dB, return_sigma=False):
    # u_ref is torch tensor!

    # change SNR in dB to SNR
    SNR = 10.0 ** (SNR_in_dB / 10.0)

    
    u_ref_flat = u_ref.flatten()
    Sq_u_ref_flat = torch.pow(u_ref_flat, 2)
    SqSigma = torch.mean(Sq_u_ref_flat)
    # Stelio's formula
    Sigma = torch.sqrt(SqSigma) / SNR
    # the "real" formula 
    # Sigma = torch.sqrt(SqSigma / SNR)

    # generate noise
    noise = torch.distributions.Normal(torch.tensor(0.0), Sigma).sample(u_ref.size())

    # add noise
    u_obs = u_ref + noise

    if return_sigma:
        # check what sigma is now
        # sigma_new = torch_funcs.std(noise)
        peak_u = torch.max(u_ref_flat)
        # return u_obs, sigma_new
        print("The noise applied of {} dB is equivalent to a SNR of {:.2f} which "
              "means sigma = {:.3g}. peak u is {:.3g}".format(SNR_in_dB, SNR, Sigma, peak_u))
        return u_obs, Sigma
    else:
        return u_obs
