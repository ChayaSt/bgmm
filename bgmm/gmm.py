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
from tqdm import tqdm


class State:
    def __init__(self, z, mu, sigma_sq_mu, sigma_sq_n, pi):
        self.z = z                     # Assignments (represented as an array with K assignment given as integer) (N)
        self.counts = np.zeros(pi.shape)               #
        self.pi = pi                   # Mixture probabilities (K)
        self.mu = mu                   # cluster means (Dim x K)
        self.sigma_sq_mu = sigma_sq_mu # Between cluster variance (hyperparameter on Gaussian prior for mu)
        self.sigma_sq_n = sigma_sq_n   # Within cluster variance


class Model:
    def __init__(self, alpha, K, sigma_sq_mu_prior, sigma_sq_n_prior, dim=1):
        self.alpha = alpha            # Parameter for Dirichlet prior over mixture probabilities (K)
        self.K = K                    # Number of components
        self.sigma_sq_mu_prior = sigma_sq_mu_prior  # Inverse Gamma
        self.sigma_sq_n_prior = sigma_sq_n_prior    # Inverse Gamma
        self.dim = dim               # dimension (default is 1)

    def cond_pi(self, state):
        counts = np.zeros(self.K)
        for i in range(self.K):
            counts[i] = len(np.where(state.z == i)[0])
        #state.counts = np.bincount(state.z)
        #if state.counts.shape < self.alpha.shape:
           #state.counts = np.append(state.counts, np.ones(1))
        state.counts = counts
        return Dirichlet(self.alpha + state.counts)

    def cond_z(self, state, X):
        # ToDo make sure the evidence is for all dimensions (that it works for multidimensional Gaussian)
        nax = np.newaxis
        prior = np.log(state.pi)
        #evidence = Gaussian(state.mu[nax, :, :], state.sigma_sq_n).log_p(X[:, nax, :]).sum(2)
        evidence = Gaussian(state.mu.reshape(self.K), state.sigma_sq_n).log_p(X)

        post = prior + evidence

        # normalize
        pvals = np.exp(post - logsumexp(post, axis=1)[:, nax])
        # C = 1.0/(np.sqrt(2*np.pi) * state.sigma_sq_n)
        # p_z_k = state.pi.reshape(3,) * (C * np.exp(-0.5*((X - state.mu.reshape(3,))/state.sigma_sq_n.reshape(3,))**2))
        # print(p_z_k)
        # #normalize
        # p_z_k /= p_z_k.sum(1)[:, np.newaxis]
        #print(pvals)
        return Multinomial(pvals)

    def cond_mu(self, state, X):
        ndata, ndim = X.shape
        h = np.zeros((self.K, ndim))
        lam = np.zeros((self.K, ndim))
        for k in range(self.K):
            idx = np.where(state.z == k)[0]
            if idx.size > 0:
                h[k, :] = X[idx, :].sum(0) / state.sigma_sq_n[k] # hyperparemter U_0 is zero
                lam[k, :] = idx.size / state.sigma_sq_n[[k]] + 1. / state.sigma_sq_mu
            else:
                h[k, :] = 0
                lam[k, :] = 1. / state.sigma_sq_mu
        return Gaussian(h/lam, 1./lam)

    def cond_sigma_sq_mu(self, state):
        ndim = state.mu.shape[1]
        a = self.sigma_sq_mu_prior.a_mu + \
            0.5 * self.K * ndim
        b = self.sigma_sq_mu_prior.b_mu + \
            0.5 * np.sum(state.mu ** 2)
        return InverseGamma(a, b)

    def cond_sigma_sq_n(self, state, X):
        # ToDo something is not working here. Try a Jefferey's prior? Or maybe do mu and sigma together?
        ndata = state.counts
        ndim = self.dim
        b = np.zeros(self.K)
        a = self.sigma_sq_n_prior.a_mu + \
            0.5 * ndata * ndim
        for k in range(self.K):
            idx = np.where(state.z == k)[0]
            b[k] = self.sigma_sq_n_prior.b_mu[k] + \
                0.5 * np.sum((X[idx] - state.mu[k]) ** 2)
        return InverseGamma(a, b)

    def cond_mu_jeff(self, state, X):
        mu = np.zeros((self.K, self.dim))
        sigma = state.sigma_sq_n.reshape(self.K, self.dim)
        for k in range(self.K):
            idx = np.where(state.z == k)[0]
            if len(idx) ==0:
                mu[k] = 0
                sigma[k] = 100
                print('Warning: No samples in component')
            if len(idx) > 0:
                mu[k] = np.mean(X[idx])
                sigma[k] /= len(idx)
        return Gaussian(mu, sigma)

    def cond_sigma_jeff(self, state, X):
        # scaled inverse chisquared with a_mu = -1/2, b_mu = 0
        idx = np.zeros(self.K)
        sigmahat2 = np.zeros(self.K)
        for k in range(self.K):
            id = np.where(state.z == k)[0]
            idx[k] = len(id)
            if idx[k] <= 1:
                idx[k] = 0.001
                sigmahat2[k] = 0.001
            else:
                idx[k] -= 1
                sigmahat2[k] = np.mean((X[id] - state.mu[k])**2)
        return InverseGamma(idx/2, (sigmahat2*idx)/2)

    def cond_mu_sigma(self, state, X):
        mu = np.zeros(self.K)
        sigma = state.sigma_sq_n
        for k in range(self.K):
            idx = np.where(state.z == k)[0]
            if len(idx) == 0:
                mu[k] = 0
                print('Warning, no samples in component')
            if len(idx) > 0:
                #mu[k] = Gaussian(np.mean(X[idx]), state.sigma_sq_n[k]/len(idx)).sample()
                mu[k] = np.random.randn()*state.sigma_sq_n[k]/len(idx) + np.mean(X[idx])
            if len(idx) > 1:
                chaisquared = np.random.chisquare(df=len(idx)-1)
                sigmahat2 = np.mean((X[idx] - mu[k])**2)
                sigma[k] = sigmahat2*len(idx)/chaisquared
                #sigma[k] = InverseGamma(a_mu=(len(idx)-1)/2, b_mu=(len(idx)-1*sigmahat2/2)).sample()
        return mu.reshape(self.K, self.dim), sigma

    def joint_log_p(self, state, X):
        return (Dirichlet(self.alpha*np.ones(self.K)).log_p(state.pi) +
                Multinomial(state.pi).log_p(state.z).sum() +
                self.sigma_sq_mu_prior.log_p(state.sigma_sq_mu) +
                self.sigma_sq_n_prior.log_p(state.sigma_sq_n).sum() +
                Gaussian(0., state.sigma_sq_mu).log_p(state.mu).sum() +
                Gaussian(state.mu[state.z, :].reshape(X.shape[0]),
                         state.sigma_sq_n.reshape(self.K, 1)[state.z, :].reshape(X.shape[0])).log_p(X.reshape(X.shape[0])).sum())

    def forward_sample(self, N, state=None):
        STATE=True
        if not state:
            STATE=False
            # Generate random state
            pi = Dirichlet(self.alpha).sample()
            mu = np.random.uniform(low=0, high=6, size=self.K).reshape(self.K, self.dim)
            sigma_sq_n = np.random.uniform(low=0, high=0.3, size=self.K)
            sigma_sq_mu = np.random.uniform(low=0, high=5)
            z = np.zeros(N)
            state = State(pi=pi, z=z, mu=mu, sigma_sq_n=sigma_sq_n, sigma_sq_mu=sigma_sq_mu)

        cdf = np.cumsum(state.pi)
        samples = np.zeros([N])
        z = np.zeros(N, dtype='int64')
        for i in range(N):
            rand = np.random.random()
            m = cdf.searchsorted(rand)
            z[i] = m
            samples[i] = Gaussian(state.mu[m], state.sigma_sq_n[m]).sample()

        if not STATE:
            # update state
            state.z = z
            state.counts = np.bincount(z)
            return state, samples
        else:
            return samples


class Sampler(object):

    def __init__(self, niter, model, state, X, prior='Jeff'):
        self.logger = {'pi': np.zeros((niter, model.K)),
                       'mean': np.zeros((niter, model.K, model.dim)),
                       'sigma_sq_mu': np.zeros((niter, 1)),
                       'sigma_sq_n': np.zeros((niter, model.K))}
        self.niter = niter
        self.state = copy.deepcopy(state)
        self.model = model
        self.X = X
        self.prior = prior

    def gibbs_step(self):
        self.state.pi = self.model.cond_pi(self.state).sample()
        self.state.z = self.model.cond_z(self.state, self.X).sample()
        if self.prior == 'Jeff':
            self.state.mu, self.state.sigma_sq_n = self.model.cond_mu_sigma(self.state, self.X)
            #self.state.mu = self.model.cond_mu_jeff(self.state, self.X).sample()
            #self.state.sigma_sq_n = self.model.cond_sigma_jeff(self.state, self.X).sample()
        else:
            self.state.mu = self.model.cond_mu(self.state, self.X).sample()
            self.state.sigma_sq_mu = self.model.cond_sigma_sq_mu(self.state).sample()
            self.state.sigma_sq_n = self.model.cond_sigma_sq_n(self.state, self.X).sample()

    def sample(self):
        for i in tqdm(range(self.niter)):
            self.gibbs_step()
            # update logger
            self.logger['pi'][i] = self.state.pi
            self.logger['mean'][i] = self.state.mu
            self.logger['sigma_sq_mu'][i] = self.state.sigma_sq_mu
            self.logger['sigma_sq_n'][i] = self.state.sigma_sq_n


# def random_model():
#     # Generate random model of default dim 1
#
#     K = np.random.randint(low=2, high=5)
#     alpha = np.random.uniform(low=-1, high=5, size=K)
#
#     sigma_sq_mu_prior = InverseGamma(a_mu=0.1, b_mu=0.1)
#     sigma_sq_n_prior = InverseGamma(a_mu=0.1, b_mu=0.1)
#
#     model = Model(alpha=alpha, K=K, sigma_sq_mu_prior=sigma_sq_mu_prior, sigma_sq_n_prior=sigma_sq_n_prior)
#     return model
