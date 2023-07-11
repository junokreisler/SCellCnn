import os, sys, errno, glob
import numpy as np
import pandas as pd
import six

import cellCnn
from cellCnn.utils import loadFCS, ftrans, mkdir_p, get_items
from cellCnn.model import CellCnn
from cellCnn.plotting import plot_results
from sklearn.metrics import roc_auc_score

# define input and output directories
WDIR = os.path.join(cellCnn.__path__[0], 'examples')
FCS_DATA_PATH = os.path.join(WDIR, 'example_FC')

# define output directory
OUTDIR = os.path.join(WDIR, 'output_test')
mkdir_p(OUTDIR)

# look at the measured markers
data_fcs = loadFCS(glob.glob(FCS_DATA_PATH + '/*.fcs')[0], transform=None, auto_comp=False)
print(data_fcs.channels)

# select the relevant markers for further analysis
markers = ['FSC-A', 'FSC-H', 'SSC-A', 'SSC-B-A', 'SSC-B-H', 'SSC-H', 'CD27', 'CD38',
           'CD127', 'HLA-DR', 'CD1c', 'CD141', 'CD45ra', 'CD16', 'CCR5', 'CD4',
           'CD11c', 'CD56', 'CD8', 'CCR7', 'IgD', 'CD3', 'IgM', 'IgG', 'CD28', 'CCR6',
           'CXCR5', 'Pd1', 'CD57', 'LIVE DEAD Blue-A', 'CD25', 'CD95', 'CXCR3', 'CD24',
           'CD20', 'CD45', 'CD11b', 'TCR gd', 'CD14', 'CD19', 'CD123', 'CD161']

marker_idx = [data_fcs.channels.index(label) for label in markers]
nmark = len(markers)

# load the sample names and corresponding labels (0: CMV-, 1: CMV+), here from a CSV file
# prior CMV infection status is obtained from the original study (Horowitz et al. 2013)
csv_file = os.path.join(WDIR, 'NK_fcs_samples_with_labels.csv')
fcs_info = np.array(pd.read_csv(csv_file, sep=','))
sample_ids = fcs_info[:, 0]
sample_labels = fcs_info[:, 1].astype(int)

# Here we randomly split the samples in training/validation/test sets.

# set random seed for reproducible results
np.random.seed(12345)

# cofactor for arcsinh transformation
cofactor = 5

# split the fcs files into training, validation and test set
group1 = np.where(sample_labels == 0)[0]
group2 = np.where(sample_labels == 1)[0]
l1, l2 = len(group1), len(group2)
ntrain_per_class = 3
ntest_group1 = l1 - ntrain_per_class
ntest_group2 = l2 - ntrain_per_class

# get the sample indices
train_idx1 = list(np.random.choice(group1, size=ntrain_per_class, replace=False))
test_idx1 = [i for i in group1 if i not in train_idx1]
train_idx2 = list(np.random.choice(group2, size=ntrain_per_class, replace=False))
test_idx2 = [i for i in group2 if i not in train_idx2]

# load the training samples
group1_list, group2_list = [], []
for idx in train_idx1:
    fname = os.path.join(FCS_DATA_PATH, sample_ids[idx])
    x_full = np.asarray(loadFCS(fname, transform=None, auto_comp=False))
    x = ftrans(x_full[:,marker_idx], cofactor)
    group1_list.append(x)

for idx in train_idx2:
    fname = os.path.join(FCS_DATA_PATH, sample_ids[idx])
    x_full = np.asarray(loadFCS(fname, transform=None, auto_comp=False))
    x = ftrans(x_full[:,marker_idx], cofactor)
    group2_list.append(x)

# load the test samples
t_group1_list, t_group2_list = [], []
test_phenotypes = []
for idx in test_idx1:
    fname = os.path.join(FCS_DATA_PATH, sample_ids[idx])
    x_full = np.asarray(loadFCS(fname, transform=None, auto_comp=False))
    x = ftrans(x_full[:,marker_idx], cofactor)
    t_group1_list.append(x)
    test_phenotypes.append(0)

for idx in test_idx2:
    fname = os.path.join(FCS_DATA_PATH, sample_ids[idx])
    x_full = np.asarray(loadFCS(fname, transform=None, auto_comp=False))
    x = ftrans(x_full[:,marker_idx], cofactor)
    t_group2_list.append(x)
    test_phenotypes.append(1)

# finally prepare training and vallidation data
cut = int(.8 * len(group1_list))
train_samples = group1_list[:cut] + group2_list[:cut]
train_phenotypes = [0] * len(group1_list[:cut]) + [1] * len(group2_list[:cut])
valid_samples = group1_list[cut:] + group2_list[cut:]
valid_phenotypes = [0] * len(group1_list[cut:]) + [1] * len(group2_list[cut:])
test_samples = t_group1_list + t_group2_list

# run a CellCnn analysis
model = CellCnn(ncell=200, nsubset=1000, verbose=0)

model.fit(train_samples=train_samples, train_phenotypes=train_phenotypes,
          valid_samples=valid_samples, valid_phenotypes=valid_phenotypes, outdir=OUTDIR)

# make predictions on the test cohort
test_pred = model.predict(test_samples)

# Each row in `test_pred` corresponds to a different sample
# and indicates the predicted class probabilities for that sample.
# Each row is a probability distribution and therefore always sums up to 1.
# Here we only have 2 classes: CMV- (1st column) and CMV+ (2nd column)

# look at the test set predictions
print('\nModel predictions:\n', test_pred)

# and the true phenotypes of the test samples
print('\nTrue phenotypes:\n', test_phenotypes)

# calculate area under the ROC curve for the test set
test_auc = roc_auc_score(test_phenotypes, test_pred[:,1])
print(test_auc)

# plot the results of the CellCnn analysis for the test samples in the output directory
_ = plot_results(model.results, test_samples, test_phenotypes,
                 markers, OUTDIR, filter_response_thres=0,
                 filter_diff_thres=0.2, group_a='CMV-', group_b='CMV+')

# In this example, we have two selected filters: filter 0 (first row) and filter 1 (second row).
# Filter 0 has high weights for the markers CD16, NKG2C and CD94
# and it is positively associated with class 1, that is CMV+ samples.

from IPython.display import Image
Image("./output_NK/consensus_filter_weights.png", width=600, height=350)

# We also see that filter 0 is more discriminative than filter 1,
# because the average cell filter response difference between CMV+
# and CMV- validation samples is higher for filter 0.

# The `filter_diff_thres` parameter (that is given as input to the `plot_results` function)
# is a threshold that defines which filters should be kept for further analysis.
# Given an array `filter_diff` of average cell filter response
# differences between classes (y-axis on the plot), sorted in decreasing order,
# we keep a filter `i, i > 0` if it holds that
# `filter_diff[i-1] - filter_diff[i] < filter_diff_thres * filter_diff[i-1]`.
# The default value is `filter_diff_thres`=0.2

Image("./output_NK/filter_response_differences.png", width=600, height=350)

# For filters selected as discriminative via the previously described step,
# we can inspect the cumulative distribution function (CDF) of the cell filter response.
# Based on this, we can pick a threshold on the cell filter response of the selected
# cell population. The default value for this threshold is `filter_response_thres`=0.

Image("./output_NK/cdf_filter_0.png", width=600, height=350)

# We can take a look at cell filter responses overlaid on a t-SNE map
# This plots can give us a hint about a suitable filter response cutoff threshold for the selected population.
# Here a reasonable value for this threshold is e.g. 0.7

Image("./output_NK/tsne_cell_response_filter_0.png")

# We plot again, now using a more stringent cutoff (filter_response_thres=0.7)
# on the cell filter response of the selected cell population.

_ = plot_results(model.results, test_samples, test_phenotypes,
                 markers, OUTDIR, filter_response_thres=0.7,
                 filter_diff_thres=0.2, group_a='CMV-', group_b='CMV+')

Image("./output_NK/cdf_filter_0.png", width=600, height=350)

# Marker abundance histograms of each selected cell population are compared
# with the corresponding marker abundance histograms of all cells. The distance between
# distributions is quantified via the Kolmogorov-Smirnov (KS) statistic.

# Here the selected cell population is now a CD57+ NKG2C+ NK cell subset

Image("./output_NK/selected_population_distribution_filter_0.png")

# For binary classification problems, we also get a boxplot of frequencies of the selected
# cell population in samples of the two classes.

Image("./output_NK/selected_population_frequencies_filter_0.png", width=300, height=175)
