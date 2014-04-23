""" 
This is a fork from parietal.probabilistic_parcellation.data_simulation.
Do not use it if you ahve access to the original one.

This module generates spatial data
to be studied in the parcellation framework

Author: Bertrand Thirion, 2013
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward, KMeans


def generate_data(mu=0, sigma1=1, sigma2=1, n_voxels=1, n_subjects=1, seed=1,
                  beta=None):
    """Generate data according to a mixed effects variance model"""
    if seed is not None:
        np.random.seed([seed])
    if beta == None:
        beta = sigma2 * np.random.randn(n_subjects)
    u = np.repeat(np.arange(n_subjects), n_voxels)
    X = np.zeros((n_voxels * n_subjects, n_subjects))
    X[np.arange(n_voxels * n_subjects), u] = 1
    y = mu + np.dot(X, beta) + sigma1 * np.random.randn(n_voxels * n_subjects)
    return y, u


def generate_data_jitter(mu=0, sigma1=1, sigma2=1, masks=[], seed=1, beta=None):
    """Generate data according to a mixed effects variance model,
    but with additional jitter"""
    n_subjects = len(masks)
    if seed is not None:
        np.random.seed([seed])
    if beta == None:
        beta = sigma2 * np.random.randn(n_subjects)
    u = np.hstack([np.ones(masks[s].sum()) * s for s in range(n_subjects)])\
        .astype(np.int)
    X = np.zeros((len(u), n_subjects))
    X[np.arange(len(u)), u] = 1
    y = mu + np.dot(X, beta) + sigma1 * np.random.randn(len(u))
    return y, u


def generate_spatial_data(shape=(40, 40), n_subjects=1, n_parcels=1, mask=None,
                          mu=None, sigma1=None, sigma2=None, model='ward',
                          seed=1, smooth=0, jitter=0., verbose=0):
    """ Generate a dataset

    Parameters
    ==========
    shape: tuple, optional
           dimensions of the spatial model (assuming a grid)
    n_subjects: int, optional, the number of subjects considered
    n_parcels: int, optional, the number of generated parcels
    mask: array of shape (shape), domain-defining binary mask
    mu: array of shape (n_parcels), the mean of the simulated parcels
    sigma1: array of shape (n_parcels),
            the first-level variance of the simulated parcels
    sigma2: array  of shape (n_parcels),
            the second-level variance of the simulated parcels
    model: string, one of ['ward, kmeans'],
           model used to generate the parcellation
    seed: int, optional, random generator seed
    smooth: float optional,
            posterior smoothing of the data
    jitter: float, optional,
            spatial jitter on the positions
    verbose: boolean, optional, verbosity mode

    Returns
    =======
    xyz: array of shape (n_voxels, 3) the coordinates of the spatial data
    label: array of shape (n_voxels) indexes defining the spatial model
    X: array of shape(n_voxels, 1), signal attached to the voxels
    """
    from scipy.ndimage import gaussian_filter
    # Create the spatial model
    if mask is None:
        mask = np.ones(np.prod(shape))
        xyz = np.indices(shape).reshape(len(shape), np.prod(shape)).T
    else:
        xyz = np.vstack(np.where(mask)).T

    if model == 'kmeans':
        spatial_model = KMeans(n_clusters=n_parcels).fit(xyz)
        label = spatial_model.labels_
    elif model == 'ward':
        connectivity = grid_to_graph(*shape, mask=mask).tocsr()
        label = Ward(n_clusters=n_parcels, connectivity=connectivity).fit(
            np.random.randn(mask.sum(), 100)).labels_
        from sklearn import neighbors
        spatial_model = neighbors.KNeighborsClassifier(3)
        spatial_model.fit(xyz, label)
    else:
        raise ValueError('%s Unknown simulation model' % model)

    if jitter > 0:
        labels = [spatial_model.predict(
                xyz + jitter * np.random.rand(1, xyz.shape[1]))
                  for subj in range(n_subjects)]

    X = np.zeros((xyz.shape[0], n_subjects))
    # Generate the functional data
    if mu == None:
        mu = np.zeros(n_parcels)
    if sigma1 == None:
        sigma1 = np.ones(n_parcels)
    if sigma2 == None:
        sigma2 = np.ones(n_parcels)
    beta_ = np.random.randn(n_subjects)

    for k in range(n_parcels):
        if jitter > 0:
            mask = [label_ == k for label_ in labels]
        else:
            mask = [label == k for subj in range(n_subjects)]
        x, subj = generate_data_jitter(mu[k], sigma1[k], sigma2[k], mask,
                                       seed=seed, beta=beta_ * sigma2[k])

        for n_subj in range(n_subjects):
            X[mask[n_subj], n_subj] = x[subj == n_subj]

    if smooth > 0:  # smooth the data
        for subj in range(n_subjects):
            X[:, subj] = gaussian_filter(
                np.reshape(X[:, subj], shape), sigma=smooth).ravel()

    if verbose:
        fig = plt.figure(figsize=(10, 1.5))
        plt.subplot(1, n_subjects + 1, 1)
        plt.imshow(np.reshape(label, shape), interpolation='nearest',
                     cmap=plt.cm.spectral)
        plt.title('Template')
        plt.axis('off')
        for ns in range(n_subjects):
            plt.subplot(1, n_subjects + 1, 2 + ns)
            plt.imshow(np.reshape(X[:, ns], shape), interpolation='nearest')
            plt.title('subject %d' % ns)
            plt.axis('off')
        plt.subplots_adjust(left=.01, bottom=.01, right=.99, top=.99,
                      wspace=.05, hspace=.01)
        fig.set_figheight(1.5)

    return xyz, label, X
