"""
Author: Chaya D. Stern
Contact: chaya.stern@choderalab.org
Date: 2016

This code is taken from Grosse and Duvenaud (Testing MCMC code)
"""


class State:
    def __init__(self, z, mu, sigma_aq_mu, sigma_sq_n, pi):
        self.z = z                     # Assignments (represented as an array of integers)
        self.mu = mu
        self.sigma_sq_mu = sigma_aq_mu # Between cluster variance
        self.sigma_sq_n = sigma_sq_n   # Withing cluster variance


class Model:
    def __init__(self, alpha, K, sigma, sigma_sq_mu_prior, sigma_sq_n_prior):
        self.alpha = alpha            # Parameter for Dirichlet prior over mixture probabilities
        self.K = K                    # Number of components
        self.sigma_sq_mu_prior = sigma_sq_mu_prior
        self.sigma_sq_n_prior = sigma_sq_n_prior

