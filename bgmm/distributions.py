"""
Author: Chaya D. Stern
Contact: chaya.stern@choderalab.org
Date: 2016

This code is taken from Grosse and Duvenaud (Testing MCMC code)
"""

import numpy as np
from scipy.special import gammaln
from scipy.stats import (dirichlet, invgamma)


class Gaussian(object):
    def __init__(self, mu, sigma_sq):
        self.mu = mu
        self.sigma_sq = sigma_sq

    def log_p(self, x):
        return -0.5 * np.log(2*np.pi) + -0.5*np.log(self.sigma_sq) + -0.5*(x-self.mu) **2 /self.sigma_sq

    def sample(self):
        return np.random.normal(self.mu, np.sqrt(self.sigma_sq))


class Dirichlet(object):
    def __init__(self, alpha):
        self.alpha = alpha

    def log_p(self, x):
        return dirichlet.logpdf(x=x, alpha=self.alpha)

    def sample(self):
        return np.random.dirichlet(self.alpha)


class InverseGamma(object):
    def __init__(self, a_mu, b_mu):
        self.a_mu = a_mu
        self.b_mu = b_mu

    def log_p(self, x):
        return invgamma.logpdf(x=x, a=self.a_mu, scale=self.b_mu)

    def sample(self):
        return self.b_mu*invgamma.rvs(self.a_mu)


class Multinomial(object):
    def __init__(self, pi, rso=np.random):
        if not np.isclose(pi.sum(1), 1.0).all():
            raise ValueError("event probability do not sum to 1")
        self.pi = pi  # can be array of pi's
        self.rso = rso

        self.logp = np.log(self.pi)

    def log_p(self, x):
        """
        ToDo: extend to array of pi's
        :param x: numpy array of length len(pi)
            The number of occurrences of each outcome
        :return: log-PMF for draw 'x'

        """
        # total number of events
        n = np.sum(x)

        # equivalent to log(n!)
        log_n_factorial = gammaln(n+1)
        # equivalent to log(x1!*...*xk!)
        sum_log_xi_factorial = np.sum(gammaln(x+1))

        log_pi_xi = self.logp*x
        log_pi_xi[ x==0 ] = 0
        # equivalent to log(p1^x1*...*pk^k)
        sum_log_pi_xi = np.sum(log_pi_xi)

        return log_n_factorial - sum_log_xi_factorial + sum_log_pi_xi

    def sample(self, n=1):
        z = np.zeros(self.pi.shape())
        for i, j in enumerate(self.pi):
            z[i] = np.random.multinomial(n, pvals=j)
        return np.where(z)[-1]

