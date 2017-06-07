"""test gmm cond probabilities"""

import unittest
from bgmm import gmm
from bgmm.distributions import (Gaussian, InverseGamma, Dirichlet, Multinomial)
import numpy as np
import copy

def random_model():
    # Default dimension of 1
    K = np.random.randint(low=2, high=5)

    alpha = np.random.uniform(low=-1, high=5, size=K)

    sigma_sq_mu_prior = InverseGamma(a_mu=0.1, b_mu=0.1)
    sigma_sq_n_prior = InverseGamma(a_mu=0.1, b_mu=0.1)

    model = gmm.Model(alpha=alpha, K=K, sigma_sq_mu_prior=sigma_sq_mu_prior, sigma_sq_n_prior=sigma_sq_n_prior)
    return model

model = random_model()
state, X = model.forward_sample(N=100)
X = X.reshape(len(X), model.dim)


def test_cond_mu():
    """test conditiona mu"""
    new_state = copy.deepcopy(state)
    new_state.mu = np.random.normal(size=model.K).reshape(model.K, model.dim)
    cond = model.cond_mu(state, X)
    assert np.allclose(cond.log_p(new_state.mu).sum() - cond.log_p(state.mu).sum(),
                       model.joint_log_p(new_state, X) - model.joint_log_p(state, X))


def test_cond_pi():
    """ Test conditional pi """
    new_state = copy.deepcopy(state)
    new_state.pi = np.random.dirichlet(model.alpha)
    cond = model.cond_pi(state)
    assert np.allclose(cond.log_p(new_state.pi) - cond.log_p(state.pi),
                       model.joint_log_p(new_state, X) - model.joint_log_p(state, X))


def test_cond_z():
    """ Test conditional z """
    new_state = copy.deepcopy(state)
    pvals = np.random.dirichlet(model.alpha, size=len(X))
    new_state.z = Multinomial(pvals).sample()
    cond = model.cond_z(state, X)
    assert np.allclose(cond.log_p(new_state.z) - cond.log_p(state.z),
                       model.joint_log_p(new_state, X) - model.joint_log_p(state, X))
