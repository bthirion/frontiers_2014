"""
This is the main script of the paper.
It perfroms a parcellation of the brain volume using various methods
and derives the average parcel signal for different contarsts.

Author: Bertrand Thirion, 2012--2014
"""
import numpy as np
import os
from nibabel import load
from nilearn import datasets
from nilearn.input_data import NiftiMasker


from group_parcellation import (
    make_parcels, reproducibility_selection, parcel_selection, 
    parcel_cv, rate_atlas)

###############################################################################
# Get the data

contrasts = ['horizontal vs vertical checkerboard',
             'left vs right button press',
             'button press vs calculation and sentence listening/reading',
             'auditory processing vs visual processing',
             'auditory&visual calculation vs sentences',
             'sentence reading vs checkerboard']

nifti_masker = NiftiMasker('mask_GM_forFunc.nii')
grp_mask = load(nifti_masker.mask).get_data()
affine = load(nifti_masker.mask).get_affine()

# Create the data matrix
n_contrasts, n_subjects = len(contrasts), 40
subjects = ['S%02d' % i for i in range(n_subjects)]
n_voxels = grp_mask.sum()
X = np.zeros((n_voxels, n_contrasts, n_subjects))

for nc, contrast in enumerate(contrasts):
    imgs = datasets.fetch_localizer_contrasts(
        [contrast], n_subjects=n_subjects)['cmaps']
    X[:, nc, :] = nifti_masker.fit_transform(imgs).T

# improve the mask
second_mask = (X == 0).sum(1).sum(1) < 100
grp_mask[grp_mask > 0] = second_mask
X = X[second_mask]

# write directory
write_dir = os.path.join(os.getcwd(), 'results')
if not os.path.exists(write_dir):
    os.mkdir(write_dir)

###############################################################################

# what shall we do in the present experiment ?
do_parcel = True
do_parcel_selection = False
do_parcel_reproducibility = False
do_parcel_cv = False
do_atlas = False
do_atlas_comparison = False

k_range = [10, 20, 30, 40, 50, 70, 100, 150, 200, 300, 400, 500, 700, 1000,
           1500, 2000, 3000, 5000, 7000, 10000]

if do_parcel_cv:
    method = 'spectral'
    print method
    llr = parcel_cv(
        X, grp_mask, write_dir=write_dir,
        method=method, n_folds=.2, k_range=k_range)

if do_parcel_selection:
    method = 'kmeans'
    print method
    llr, bic = parcel_selection(
        X, grp_mask, write_dir=write_dir, method=method, k_range=k_range, 
        criterion='ll')
  
if do_parcel: 
    for method in ['ward', 'kmeans', 'geometric', 'spectral']:
        print method
        ll, bic = make_parcels(
            X, grp_mask, contrasts, affine, subjects,
            write_dir=write_dir, method=method, n_clusters=158)
        print ll, bic

if do_atlas:
    atlases = ['sri24_3mm.nii', 'HarvardOxford-labels-3mm-slpit.nii']
    for atlas in atlases:
        labels = load(atlas).get_data()[grp_mask]
        rate_atlas(X, labels, write_dir=write_dir)

if do_parcel_reproducibility:
    method = 'spectral'
    print method
    r_score = reproducibility_selection(
        X, grp_mask, niter=5, method=method, k_range=k_range, 
        write_dir=write_dir)

if do_atlas_comparison:
    atlas = load('HarvardOxford-labels-3mm-slpit.nii')
    labels = atlas.get_data()[grp_mask]
    ll = {'ward':[], 'kmeans':[], 'geometric':[], 'spectral':[], 'atlas':[]}
    for i in range(30):
        bootstrap_sample = (
            n_subjects * np.random.rand(n_subjects)).astype(np.int)
        X_ = X[:, :, bootstrap_sample]
        for method in ['ward', 'kmeans', 'geometric', 'spectral']:
            print method
            ll_, bic = make_parcels(
                X_, grp_mask, contrasts, affine, subjects,
                write_dir=write_dir, method=method, n_clusters=158)
            ll[method].append(ll_)

        ll_, _ = rate_atlas(X_, labels, write_dir=write_dir)
        ll['atlas'].append(ll_)


