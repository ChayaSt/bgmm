"""
Author: Chaya D. Stern
Contact: chaya.stern@choderalab.org
Date: 2016

This code is taken from Grosse and Duvenaud (Testing MCMC code)
Meanwhile it only has isotropic sigma.
ToDo: extend to other variance
"""

import numpy as np
from bgmm.distributions import (Dirichlet, Multinomial, Gaussian, InverseGamma)
import copy
from scipy.misc import logsumexp


class State:
    def __init__(self, z, mu, sigma_aq_mu, sigma_sq_n, pi):
        self.z = z                     # Assignments (represented as an array with K assignment given as integer) (N)
        self.pi = pi                   # Mixture probabilities (K)
        self.mu = mu                   # cluster means (Dim x K)
        self.sigma_sq_mu = sigma_aq_mu # Between cluster variance
        self.sigma_sq_n = sigma_sq_n   # Within cluster variance


class Model:
    def __init__(self, alpha, K, sigma_sq_mu_prior, sigma_sq_n_prior, dim=1):
        self.alpha = alpha            # Parameter for Dirichlet prior over mixture probabilities (Dim x K)
        self.K = K                    # Number of components
        self.sigma_sq_mu_prior = sigma_sq_mu_prior  # Inverse Gamma
        self.sigma_sq_n_prior = sigma_sq_n_prior    # Inverse Gamma
        self.dim = dim               # dimension (default is 1)

    def cond_pi(self, state):
        counts = np.bincount(state.z)
        return Dirichlet(self.alpha + counts)

    def cond_z(self, state, X):
        # ToDo make sure the evidence is for all dimensions (that it works for multidimensional Gaussian)
        nax = np.newaxis
        prior = np.log(state.pi)
        evidence = Gaussian(state.mu, state.sigma_sq_n).log_p(X)
        post = prior + evidence

        # normalize
        pvals = np.exp(post - logsumexp(post, axis=1)[:, nax])
        return Multinomial(pvals)

    def cond_mu(self, state, X):
        ndata, ndim = X.shape
        h = np.zeros((self.K, ndim))
        lam = np.zeros((self.K, ndim))
        for k in range(self.K):
            idx = np.where(state.z == k)[0]
            if idx.size > 0:
                h[k, :] = X[idx, :].sum(0) / state.sigma_sq_n[k]
                lam[k, :] = idx.size / state.sigma_sq_n[k] +1. / state.sigma_sq_mu[k]
            else:
                h[k, :] = 0
                lam[k, :] = 1. / state.sigma_sq_mu[k]
        return Gaussian(h/lam, 1./lam)

    def cond_sigma_sq_mu(self, state):
        ndim = state.mu.shape[1]
        a = self.sigma_sq_mu_prior.a + \
            0.5 * self.K * ndim
        b = self.sigma_sq_mu_prior.b + \
            0.5 * np.sum(state.mu ** 2)
        return InverseGamma(a, b)

    def cond_sigma_sq_n(self, state, X):
        ndata, ndim = X.shape
        a = self.sigma_sq_n_prior.a + \
            0.5 * ndata * ndim
        b = self.sigma_sq_n_prior.b + \
            0.5 * np.sum((X - state.mu[state.z, :]) ** 2)
        return InverseGamma(a, b)

    def joint_log_p(self, state, X):
        return (Dirichlet(self.alpha*np.ones(self.K)).log_p(state.pi) +
                Multinomial(self.state.pi).log_p(state.z).sum() +
                self.sigma_sq_mu_prior.log_p(state.sigma_sq_mu) +
                self.sigma_sq_n_prior.log_p(state.sigma_sq_n) +
                Gaussian(0., state.sigma_sq_mu).log_p(state.mu).sum() +
                Gaussian(state.mu[state.z, :], state.sigma_sq_n).log_p(X).sum())


class Sampler(object):

    def __init__(self, niter, model, state, X):
        self.logger = {'pi': np.zeros((niter, state.pi.shape)),
                       'mean': np.zeros((niter, state.mu.shape)),
                       'sigma_sq_mu': np.zeros((niter, state.sigma_sq_mu.shape)),
                       'sigma_sq_n': np.zeros((niter, state.sigma_sq_n.shape))}
        self.niter = niter
        self.state = copy.deepcopy(state)
        self.model = model
        self.X = X

    def gibbs_step(self):
        self.state.pi = self.cond_pi(self.state).sample()
        self.state.z = self.cond_z(self.state, self.X).sample()
        self.state.mu = self.cond_mu(self.state, self.X).sample()
        self.state.sigma_sq_mu = self.cond_sigma_sq_mu(self.state).sample()
        self.state.sigma_sq_n = self.cond_sigma_sq_n(self.state, self.X).sample()

    def sample(self):
        for i in range(self.niter):
            self.model.gibbs_step(self.state, self.X)
            # update logger
            self.logger['pi'][i] = self.state.pi
            self.logger['mean'][i] = self.state.mu
            self.logger['sigma_sq_mu'] = self.state.sigma_sq_mu
            self.logger['sigma_sq_n'] = self.state.sigma_sq_n
