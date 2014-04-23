"""
This is a fork from parietal.probabilistic_parcellation/group_parcellation.py
Do not use it if you have access to the original

This module contains some code to perform group analysis.
It is derived to fit the needs that appear on a random basis, 
that are shared across studies:
parcellation, blobs on random effects, mixed effects, functional landmarks...

Author: Bertrand Thirion, 2012-2013
"""
from os import path
import hashlib
import csv
import pickle

from scipy.sparse import dia_matrix
import numpy as np
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import Ward, spectral_clustering, k_means
from sklearn.decomposition import PCA
from sklearn.manifold import spectral_embedding
from sklearn.cluster.spectral import discretize

from nibabel import save, Nifti1Image
from nipy.labs import viz

from mixed_effects_parcel import (
    parameter_map, reproducibility_rating, score_spatial_model,
    log_likelihood_map)

KRANGE = [100, 200, 500]


def parcel_cv(X, grp_mask, write_dir='/tmp/', method='ward', n_folds=10, 
              k_range=KRANGE, verbose=True):
    """ Functiond edicated to parcel selection using 10-fold cross-validation"""
    from sklearn.cross_validation import KFold, ShuffleSplit
    # Define the structure A of the data. Pixels connected to their neighbors.
    n_voxels, n_contrasts, n_subjects = X.shape
    n_components = 100

    # Define a spatial model
    shape = grp_mask.shape
    connectivity = grid_to_graph(shape[0], shape[1], shape[2], grp_mask).tocsr()
    ic, jc = connectivity.nonzero()

    # concatenate the data spatially
    Xv = np.reshape(X, (n_voxels, n_contrasts * n_subjects))
    sigma = np.sum((Xv[ic] - Xv[jc]) ** 2, 1).mean()

    # pre-compute PCA for the cross_validation loops
    if n_folds == int(n_folds):
        cv = KFold(X.shape[2], n_folds)
    else:
        cv = ShuffleSplit(X.shape[2], 10, .2)
    maps = []
    for (train, test) in cv:
        X_ = np.reshape(X[:, :, train], (n_voxels, n_contrasts * len(train)))
        
        if method == 'spectral':
            connectivity.data = np.exp( 
                - np.sum((X_[ic] - X_[jc]) ** 2, 1) / (2 * sigma))
            maps.append(spectral_embedding(
                    connectivity, n_components=n_components,
                    eigen_solver='arpack', random_state=None,
                    eigen_tol=0.0, drop_first=False))
        else:
            maps.append(PCA(n_components=n_components).fit_transform(X_))

    # parcel selection
    all_crit = {}
    for k in k_range:
        ll, ll_cv = 0, 0
        for (it, (train, test)) in enumerate(cv):
            if method == 'ward':
                ward = Ward(n_clusters=k, 
                            connectivity=connectivity).fit(maps[it])
                labels = ward.labels_
            elif method in ['k-means', 'kmeans']:
                _, labels, _ = k_means(maps[it], n_clusters=k, n_init=1,
                         precompute_distances=False, max_iter=10)
            elif method == 'spectral':
                 if k <= n_components:
                     for i in range(10):
                         labels = discretize(maps[it][:, :k])
                         if len(np.unique(labels)) == k:
                             break
                 else:
                     _, labels, _ = k_means(
                         maps[it], n_clusters=k, n_init=1,
                         precompute_distances=False, max_iter=10)
            elif method == 'geometric':
                xyz = np.array(np.where(grp_mask)).T
                _, labels, _ = k_means(xyz, n_clusters=k, n_init=1,
                                       precompute_distances=False, max_iter=10)
            for contrast in range(n_contrasts):
                ll1, mu_, sigma1_, sigma2_, bic_ = parameter_map(
                    X[:, contrast, train], labels, null=False)
                ll += ll1.sum()
                ll2 = log_likelihood_map(
                    X[:, contrast, test], labels, mu_, sigma1_, sigma2_)

                ll_cv += ll2.sum()
        all_crit[k] = ll_cv
        if verbose:
            print 'k: ', k, 'll: ', ll, ' ll_cv: ', ll_cv
    
    file = open(path.join( write_dir, 'll_cv_%s.pck' % method), 'w')
    pickle.dump(all_crit, file)
    return all_crit


def rate_atlas(X, labels, write_dir='/tmp/', criterion='ll', method='atlas', 
               verbose=True):
    """Yield the scores obtained by a given atlas"""
    n_voxels, n_contrasts, n_subjects = X.shape
    ll, bic = 0, 0
    # remove absent labels
    labels_ = np.asarray(labels).copy()
    relabel = np.zeros(labels.max() + 1)
    relabel[np.array([x for x in np.unique(labels)if x != -1])] = 1
    relabel = (np.cumsum(relabel) - 1).astype(np.int16)
    labels_[labels > -1] = relabel[labels[labels > -1]]

    for contrast in range(n_contrasts):
        ll1, mu_, sigma1_, sigma2_, bic_ = parameter_map(
            X[:, contrast], labels_, null=False)
        bic += bic_.sum()
        if criterion == 'log-LR':
            ll2, _, _, _, bic_ = parameter_map(
                X[:, contrast], labels_, null=True)
            ll += np.sum((ll1 - ll2))
        elif criterion == 'll':
            ll += np.sum(ll1)
        elif criterion == 'sigma':
            ll = (sigma1_.mean(), sigma2_.mean())
        all_bic = bic
        all_crit = ll
    if verbose:
        print ' bic: ', bic, ' crit: ', ll
    if criterion == 'log-LR':
        file = open(path.join( write_dir, 'all_llr_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    elif criterion == 'll':
        file = open(path.join( write_dir, 'all_ll_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    elif criterion == 'sigma':
        file = open(path.join( write_dir, 'all_sigma_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    elif criterion == 'kfold':
        file = open(path.join( write_dir, 'all_kfold_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    file = open(path.join( write_dir, 'all_bic_%s.pck' % method), 'w')
    pickle.dump(all_bic, file)
    return all_crit, all_bic


def parcel_selection(X, grp_mask, write_dir='/tmp/', method='ward',
                     k_range=KRANGE, criterion='ll', verbose=True):
    """ Functiond edicated to parcel selection """
    # Define the structure A of the data. Pixels connected to their neighbors.
    n_voxels, n_contrasts, n_subjects = X.shape
    n_components = 100

    # Define a spatial model
    shape = grp_mask.shape
    connectivity = grid_to_graph(shape[0], shape[1], shape[2], grp_mask).tocsr()

    # concatenate the data spatially
    Xv = np.reshape(X, (n_voxels, n_contrasts * n_subjects))
    X_ = PCA(n_components=n_components).fit_transform(Xv)

    i, j = connectivity.nonzero()
    sigma = np.sum((Xv[i] - Xv[j]) ** 2, 1).mean()
    if method == 'spectral':
        i, j = connectivity.nonzero()
        sigma = np.sum((Xv[i] - Xv[j]) ** 2, 1).mean()
        connectivity.data = np.exp( - np.sum((Xv[i] - Xv[j]) ** 2, 1) /
                                      (2 * sigma))
        
        maps = spectral_embedding(connectivity, n_components=n_components,
                              eigen_solver='arpack',
                              random_state=None,
                              eigen_tol=0.0, drop_first=False)
        
    del Xv
   
    # parcel selection
    all_bic = {}
    all_crit = {}
    for k in k_range:
        if method == 'ward':
            ward = Ward(n_clusters=k, 
                        connectivity=connectivity).fit(X_)
            labels = ward.labels_
        elif method == 'spectral':
            if k <= n_components:
                for i in range(10):
                    labels = discretize(maps[:, :k])
                    if len(np.unique(labels)) == k:
                        break
            else:
                _, labels, _ = k_means(maps[:, :100], n_clusters=k, n_init=1,
                         precompute_distances=False, max_iter=10)
        elif method == 'geometric':
            xyz = np.array(np.where(grp_mask)).T
            _, labels, _ = k_means(xyz, n_clusters=k, n_init=1,
                                   precompute_distances=False, max_iter=10)
        elif method in ['k-means', 'kmeans']:                
            _, labels, _ = k_means(X_, n_clusters=k, n_init=1,
                                   precompute_distances=False, max_iter=10)
        elif method == 'gmm':
            from sklearn.mixture import GMM
            labels = GMM(n_components=k, covariance_type='spherical', n_iter=10,
                      n_init=1).fit(X_).predict(X_)
            
        ll, bic = 0, 0
        for contrast in range(n_contrasts):
            ll1, mu_, sigma1_, sigma2_, bic_ = parameter_map(
                X[:, contrast], labels, null=False)
            bic += bic_.sum()
            if criterion == 'log-LR':
                ll2, _, _, _, bic_ = parameter_map(
                    X[:, contrast], labels, null=True)
                ll += np.sum((ll1 - ll2))
            elif criterion == 'll':
                ll += np.sum(ll1)
            elif criterion == 'sigma':
                ll = (sigma1_.mean(), sigma2_.mean())
            elif criterion == 'kfold':
                ll += score_spatial_model(X[:, contrast], labels, cv='kfold')
        all_crit[k] = ll
        all_bic[k] = bic
        if verbose:
            print 'k: ', k, ' bic: ', bic, ' crit: ', ll
    if criterion == 'log-LR':
        file = open(path.join( write_dir, 'all_llr_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    elif criterion == 'll':
        file = open(path.join( write_dir, 'all_ll_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    elif criterion == 'sigma':
        file = open(path.join( write_dir, 'all_sigma_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    elif criterion == 'kfold':
        file = open(path.join( write_dir, 'all_kfold_%s.pck' % method), 'w')
        pickle.dump(all_crit, file)
    file = open(path.join( write_dir, 'all_bic_%s.pck' % method), 'w')
    pickle.dump(all_bic, file)
    return all_crit, all_bic


def reproducibility_selection(
    X, grp_mask, niter=2, method='ward', k_range=KRANGE, write_dir='/tmp',
    verbose=True):
    """ Returns a reproducibility metric on bootstraped models
    
    Parameters
    ----------
    X: array of shape (n_voxels, n_contrasts, n_subjects)
       the input data
    grp_mask: array of shape (image_shape),
              the non-zeros elements yield the spatial model
    niter: int, number of bootstrap samples estimated
    method: string, one of 'ward', 'kmeans', 'spectral'
    k_range: list of ints, 
             the possible number of parcels to be tested
    """
    n_voxels, n_contrasts, n_subjects = X.shape
    n_components = 100

    # Define a spatial model
    shape = grp_mask.shape
    connectivity = grid_to_graph(shape[0], shape[1], shape[2], grp_mask).tocsr()
    # concatenate the data spatially
    Xv = np.reshape(X, (n_voxels, n_contrasts * n_subjects))
    # pre-computed stuff
    ic, jc = connectivity.nonzero()
    sigma = np.sum((Xv[ic] - Xv[jc]) ** 2, 1).mean()
    
    maps = []
    for i in range(niter):
        bootstrap = (np.random.rand(Xv.shape[1]) * Xv.shape[1]).astype(int)
        X_ = Xv[:, bootstrap]
        if method == 'spectral':
            connectivity.data = np.exp( 
                - np.sum((X_[ic] - X_[jc]) ** 2, 1) / (2 * sigma))
            maps.append(spectral_embedding(connectivity,
                                           n_components=n_components,
                                           eigen_solver='arpack',
                                           random_state=None,
                                           eigen_tol=0.0, drop_first=False))
        else:
            maps.append(PCA(n_components=n_components).fit_transform(X_))
            
    ars_score = {}
    ami_score = {}
    vm_score = {}
    for (ik, k_) in enumerate(k_range):
        label_ = []
        for i in range(niter):
            bootstrap = (np.random.rand(Xv.shape[1]) * Xv.shape[1]).astype(int)
            if method == 'spectral':
                if k_ <= n_components:
                    for _ in range(10):
                        labels = discretize(maps[i][:, :k_])
                        if len(np.unique(labels)) == k_:
                            break
                else:
                    _, labels, _ = k_means(
                        maps[i], n_clusters=k_, n_init=1,
                        precompute_distances=False, max_iter=10)
            elif method == 'ward':
                    ward = Ward(n_clusters=k_, 
                                connectivity=connectivity).fit(maps[i])
                    labels = ward.labels_
            elif method in ['k-means', 'kmeans']:
                _, labels, _ = k_means(maps[i], n_clusters=k_, n_init=1,
                                       precompute_distances=False, max_iter=10)
            elif method == 'geometric':
                xyz = np.array(np.where(grp_mask)).T
                _, labels, _ = k_means(xyz, n_clusters=k_, n_init=1,
                                       precompute_distances=False, max_iter=10)
            label_.append(labels)
        ars_score[k_] = reproducibility_rating(label_, 'ars')
        ami_score[k_] = reproducibility_rating(label_, 'ami')
        vm_score[k_] = reproducibility_rating(label_, 'vm')
        if verbose:
            print 'k: ', k_, '  ari: ', ars_score[k_], 'ami: ',ami_score[k_],\
                ' vm: ', vm_score[k_]
    file = open(path.join(write_dir, 'ari_score_%s.pck' % method), 'w')
    pickle.dump(ars_score, file)
    file = open(path.join(write_dir, 'ami_score_%s.pck' % method), 'w')
    pickle.dump(ami_score, file)
    file = open(path.join(write_dir, 'vm_score_%s.pck' % method), 'w')
    pickle.dump(vm_score, file)
    return ars_score, ami_score, vm_score    


def make_parcels(X, grp_mask, contrasts, affine, subjects, write_dir='/tmp/',
                 method='ward', n_clusters=500, do_ttest=False,
                 do_ftest=False, do_csv=False, write_mean=False):
    # Define the structure A of the data. Pixels connected to their neighbors.
    n_voxels, n_contrasts, n_subjects = X.shape
    if len(contrasts) != n_contrasts:
        raise ValueError('Incorrect Number of contrasts provided')

    # Define a spatial model
    shape = grp_mask.shape
    connectivity = grid_to_graph(shape[0], shape[1], shape[2], grp_mask).tocsr()

    # concatenate the data spatially
    Xv = np.reshape(X, (n_voxels, n_contrasts * n_subjects))
    X_ = PCA(n_components=100).fit_transform(Xv)

    if method == 'spectral':
        i, j = connectivity.nonzero()
        sigma = np.sum((Xv[i] - Xv[j]) ** 2, 1).mean()
        connectivity.data = np.exp( - np.sum((Xv[i] - Xv[j]) ** 2, 1) /
                                      (2 * sigma))
        connectivity = connectivity.copy() + dia_matrix(
            (1.e-3 * np.ones(n_voxels), [0]), 
            shape=(n_voxels, n_voxels)).tocsr()

    # Compute clustering
    print "Compute structured hierarchical clustering..."
    if method == 'ward':
        ward = Ward(n_clusters=n_clusters, connectivity=connectivity).fit(X_)
        labels = ward.labels_
    elif method == 'spectral':
        labels = spectral_clustering(connectivity, n_clusters=n_clusters,
                                     eigen_solver='arpack', n_init=5)
    elif method in ['k-means', 'kmeans']:
        _, labels, _ = k_means(X_, n_clusters=n_clusters, n_init=5,
                               precompute_distances=False, max_iter=30)
    else:
        xyz = np.array(np.where(grp_mask)).T
        _, labels, _ = k_means(xyz, n_clusters=n_clusters, n_init=1,
                               precompute_distances=False, max_iter=10)
    wlabel = grp_mask.astype(np.int16) - 1
    wlabel[wlabel == 0] = labels
    save(Nifti1Image(wlabel, affine), path.join(
            write_dir, 'parcel_%s_%d.nii' % (method, n_clusters)))

    ll, bic = 0, 0
    for c, contrast in enumerate(contrasts):
        mu_map = np.zeros_like(wlabel).astype(np.float)
        s1_map = np.zeros_like(wlabel).astype(np.float)
        s2_map = np.zeros_like(wlabel).astype(np.float)

        ll_, mu_, sigma1_, sigma2_, bic_ = parameter_map(
            X[:, c], labels, null=False)
        ll += ll_.sum()
        bic += bic_.sum()
        if write_mean:    
            mu_map[grp_mask == 1] = mu_[labels]
            s1_map[grp_mask == 1] = sigma1_[labels]
            s2_map[grp_mask == 1] = sigma2_[labels]
            save(Nifti1Image(mu_map, affine), path.join(write_dir, 'mu_%s.nii' %
                                                        contrast))
            save(Nifti1Image(s1_map, affine), path.join(write_dir, 's1_%s.nii' %
                                                        contrast))
            save(Nifti1Image(s2_map, affine), path.join(write_dir, 's2_%s.nii' %
                                                        contrast))

    # Get the signals per parcel
    mean_X = np.empty((n_clusters, n_contrasts, n_subjects), np.float)
    for k in range(n_clusters):
        mean_X[k] = X[labels == k].mean(0).reshape(
            n_subjects, n_contrasts).T

    if do_ttest:
        # create one-sample t-tests images
        wlabel[grp_mask == 1] = labels
        active = np.array(np.maximum(0, wlabel.astype(np.float)))
        for c, contrast in enumerate(contrasts):
            t_test = mean_X[:, c].mean(1) / mean_X[:, c].std(1) *\
                np.sqrt(n_subjects)
            active[grp_mask == 1] = t_test[(labels).astype(np.int16)]
            viz.plot_map(active, affine, threshold=4.0, cmap=viz.cm.cold_hot, 
                         vmin=-20., vmax=20)

    if do_ftest:
        # pseudo F-test
        F_test = n_subjects * (mean_X.mean(2) ** 2  / mean_X.var(2)).sum(1) / 3.
        active[grp_mask == 1] = F_test[(labels).astype(np.int16)]
        viz.plot_map(active, affine, threshold=4.0, cmap=viz.cm.cold_hot, 
                     vmin=-20., vmax=20)
        save(Nifti1Image(active, affine), path.join(write_dir, 'F_RFX.nii'))

    if do_csv:
        # write parcel signals as csv file
        hash_ = hashlib.sha224(wlabel).hexdigest()
        for c, contrast in enumerate(contrasts):
            wpath = path.join(write_dir, 'contrast_%s_%s.csv' % (
                    contrast, hash_))
            fid = open(wpath, 'wb')
            writer = csv.writer(fid, delimiter=' ')
            writer.writerow(subjects)
            pdata = mean_X[:, c]

            # write pdata
            for row in pdata:
                writer.writerow(row)
            fid.close()
    return ll, bic
