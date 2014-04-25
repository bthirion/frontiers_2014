"""Test the mixed_effects_parcellation module

Author: Bertrand Thirion, 2012
"""
import numpy as np
from numpy.testing import assert_almost_equal
from nose.tools import assert_true, assert_equal
from numpy.testing import assert_array_equal
from mixed_effects_parcel import (
    log_likelihood, log_likelihood_,
    cluster_spatial_data, score_spatial_model, manual_optimization, 
    cluster_selection, em_inference, em_inference_fast)
from data_simulation import (
    generate_data, generate_spatial_data)
    

def u2J(u):
    """Small function to convert a labelling in a blockwise covariance"""
    X = np.zeros((u.size, len(np.unique(u))))
    X[np.arange(u.size), u] = 1
    J = np.dot(X, X.T)
    return J

def test_generate_data():
    """Test data generator"""
    y, u = generate_data(mu=0, sigma1=1, n_subjects=1, n_voxels=30)
    assert_true(y.shape == u.shape)
    assert_true(y.var() < 2)
    assert_true( y.var() > 1)
    assert_true(y.mean() ** 2 < 1)
    assert_array_equal(u, np.zeros(u.size))

def test_em_inference():
    """ """
    y, u = generate_data(mu=0, sigma1=1, sigma2=1, n_subjects=10, n_voxels=10)
    mu, s1, s2, ll = em_inference(y, u)
    assert_almost_equal(mu, 0, 0)
    assert_almost_equal(s1, 1, 0)
    assert_almost_equal(s2, 1, 0)
    mu_, s1_, s2_, ll_ = em_inference_fast(y, u)
    assert_almost_equal(mu, mu_, 6)
    assert_almost_equal(s1, s1_, 6)
    assert_almost_equal(s2, s2_, 6)
    assert_almost_equal(ll, ll_, 6)

def test_likelihood():
    """Test that the returned likelihood is reasonable"""
    mu, sigma1, sigma2, n_subjects, n_voxels= 5., 1., 0., 1, 30
    y, u = generate_data(mu=mu, sigma1=sigma1, sigma2=sigma2, 
                          n_subjects=n_subjects, n_voxels=n_voxels)
    ll = log_likelihood(y, mu, sigma1, 0., np.eye(n_voxels)) / y.size
    h = - 0.5 * (1 + np.log(2 * np.pi))
    assert_almost_equal(ll, h, 0)
    sigma1 = 10.
    y, u = generate_data(mu=mu, sigma1=sigma1, sigma2=sigma2, 
                          n_subjects=n_subjects, n_voxels=n_voxels)
    ll = log_likelihood(y, mu, sigma1, 0., np.eye(n_voxels)) / y.size
    assert_almost_equal(ll, h - np.log(sigma1), 0)
    sigma1, sigma2, n_subjects, n_voxels = 0, 1, 30, 1
    y, u = generate_data(mu=mu, sigma1=sigma1, sigma2=sigma2, 
                          n_subjects=n_subjects, n_voxels=n_voxels)
    ll = log_likelihood(y, mu, 0., sigma2, np.eye(n_subjects)) / y.size
    assert_almost_equal(ll, h, 0)
    

def test_likelihood_u_J():
    """Test that the 2 likelihood estimators give the same result"""
    # 3 different parameter settings to test numerical stability
    
    # setting 1
    mu, sigma1, sigma2, n_subjects, n_voxels= 5., 1., 10., 5, 10
    y, u = generate_data(mu=mu, sigma1=sigma1, sigma2=sigma2, 
                          n_subjects=n_subjects, n_voxels=n_voxels)
    J = u2J(u)
    ll1 = log_likelihood(y, mu, sigma1, sigma2, J)
    ll2 = log_likelihood_(y, mu, sigma1, sigma2, u)
    assert_almost_equal(ll1, ll2)
    
    # setting2
    mu, sigma1, sigma2, n_subjects, n_voxels= -5., 10., 1., 10, 5
    y, u = generate_data(mu=mu, sigma1=sigma1, sigma2=sigma2, 
                          n_subjects=n_subjects, n_voxels=n_voxels)
    J = u2J(u)
    ll1 = log_likelihood(y, mu, sigma1, sigma2, J)
    ll2 = log_likelihood_(y, mu, sigma1, sigma2, u)
    assert_almost_equal(ll1, ll2)
    
    # setting 3
    mu, sigma1, sigma2, n_subjects, n_voxels= -5., 1.e-6, 1, 10, 1
    y, u = generate_data(mu=mu, sigma1=sigma1, sigma2=sigma2, 
                          n_subjects=n_subjects, n_voxels=n_voxels)
    J = u2J(u)
    ll1 = log_likelihood(y, mu, sigma1, sigma2, J)
    ll2 = log_likelihood_(y, mu, sigma1, sigma2, u)
    assert_almost_equal(ll1, ll2, 3)


def test_generate_spatial_data():
    shape = (40, 40)
    xyz, label, X = generate_spatial_data(shape=shape)
    assert_equal(xyz.shape, (np.prod(shape), 2))
    assert_true(label.shape == np.prod(shape))
    assert_true(X.shape == (np.prod(shape), 1))
    k, mu = 5, 10.
    xyz, label, X = generate_spatial_data(n_parcels=k, shape=shape)
    assert_equal(len(np.unique(label)), k)
    xyz, label, X = generate_spatial_data(n_parcels=5, mu=mu*np.ones(k))
    assert_almost_equal(X.mean(), mu, -1)
    

def test_cluster_spatial():
    # dot it based on the shape
    shape, k = (20, 20), 5
    xyz, _, X = generate_spatial_data(shape=shape)
    u = cluster_spatial_data(X, n_parcels=k, xyz=xyz, shape=shape)
    assert_true(len(np.unique(u)) == k)
    
    # now dot it with a mask
    mask = np.random.rand(*shape) > .1
    xyz, _, X = generate_spatial_data(shape=shape, mask=mask)
    u = cluster_spatial_data(X, n_parcels=k, xyz=xyz, shape=shape, mask=mask)
    assert_true(len(np.unique(u)) == k)
    
    # now do it with the coordinates
    u = cluster_spatial_data(X, n_parcels=k, xyz=xyz)
    assert_true(len(np.unique(u)) == k)
    

def test_score_model():
    shape, k, n_subjects = (20, 20), 5, 4
    xyz, _, X = generate_spatial_data(shape=shape, n_subjects=n_subjects)
    u = cluster_spatial_data(X, n_parcels=k, xyz=xyz, shape=shape)
    # using ll
    ll = score_spatial_model(X, u, cv=None) / (n_subjects * np.prod(shape))
    assert_almost_equal(ll, -0.5 * (1 + np.log(2 * np.pi)), 0)
    # using loo
    ll = score_spatial_model(X, u, cv='loo') / (n_subjects * np.prod(shape))
    assert_almost_equal(ll, -0.5 * (1 + np.log(2 * np.pi)), 0)
    # using kfold
    ll = score_spatial_model(X, u, cv='kfold') / (n_subjects * np.prod(shape))
    assert_almost_equal(ll, -0.5 * (1 + np.log(2 * np.pi)), 0)

def test_manual_opt():
    n_subjects, n_voxels = 5, 20
    mu, sigma1, sigma2 = 3., 10., 1.
    y, u = generate_data(n_subjects=n_subjects, n_voxels=n_voxels, mu=mu, 
                          sigma1=sigma1, sigma2=sigma2)
    mu_, sigma1_, sigma2_, ll_ = manual_optimization(y, u)
    assert_almost_equal(mu, mu_, 0)
    assert_almost_equal(sigma1, sigma1_, 0)
    assert_almost_equal(sigma2, sigma2_, 0)


def test_cluster_selection():
    n_parcels, n_subjects = 2, 6
    shape, verbose = (15, 10), False
    mu = 10 * np.random.randn(n_parcels)
    sigma2 = np.ones(n_parcels)
    xyz, label, X = generate_spatial_data(
        n_parcels=n_parcels, mu=mu, sigma2=sigma2, n_subjects=n_subjects, 
        shape=shape, seed=1)
    krange = range(1, 4)
    label, ll = cluster_selection(X, krange, shape=shape)
    n_labels = len(np.unique(label))
    assert_true(n_labels > 1)
    

if __name__ == '__main__':
    import nose
    nose.run(argv=['', __file__])
