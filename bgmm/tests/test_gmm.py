"""test gmm cond probabilities"""

import unittest
from bgmm import gmm
from bgmm.distributions import (Gaussian, InverseGamma, Dirichlet, Multinomial)
import numpy as np

def random_model():
    dim = np.random.randint(low=1, high=3)
    K = np.random.randint(low=2, high=5)

    alpha = np.random.uniform(low=-1, high=5, size=K)
    pi = np.random.dirichlet(alpha)


    sigma_sq_mu_prior = InverseGamma(a_mu=, b_mu=)
    sigma_sq_n_prior = InverseGamma(a_mu=, b_mu=)

    model = gmm.Model(alpha=alpha, K=K, sigma_sq_mu_prior, sigma_sq_n_prior)
    return model

def initialize_state(model):
    #ToDo extend to higher dimensions
    mu = np.zeros((model.K, model.dim))
    for i in range(model.K):
        mu[i] = np.random.uniform(low=-2, high=2, size=model.dim)
    sigma_sq_mu = InverseGamma(a_mu=, b_mu=).sample()
    sigma_sq_n = InverseGamma(a_mu=, b_mu=).sample()
    pi = np.random.dirichlet(model.alpha)

    # generate Z from pi
    cdf = np.cumsum(pi)
    z = np.zeros(100)
    for i in range(100):
        rand = np.random.random()
        z[i] = cdf.searchsorted(rand)

    state = gmm.State(z=z, pi=pi, mu=mu, sigma_sq_mu=sigma_sq_mu, sigma_sq_n=sigma_sq_n)
    return state

def generate_samples(state, dim):
    samples = np.zeros((100, dim))
    cdf = np.cumsum(state.pi)
    for i in range(samples.shape[0]):
        rand = np.random.random()
        m = cdf.searchsorted(rand)
        #ToDo generate greater than 1D samples
        samples[i] = state.sigma_sq_n*np.random.randn() + state.mu[m]

    return samples