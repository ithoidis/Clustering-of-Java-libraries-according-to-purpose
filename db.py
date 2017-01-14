import numpy as np

from preprocessDataset import freq, attribute, project, category
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import sys
import scipy.spatial.distance as sp
import matplotlib.pyplot as plt
##############################################################################

X, y = freq, category
labels_true = y

# Create precomputed cosine distance matrix for use in DBSCAN
##print X.shape
precomputed_mat = sp.pdist(X, 'cosine')  # find the cosine distance vector
##print precomputed_mat.shape
precomputed_mat = sp.squareform(precomputed_mat)  # make it a square mat
##print precomputed_mat.shape
##############################################################################
# Compute DBSCAN
epsilon = 0.6  # DBSCAN eps
samples = 3  # DBSCAN min_samples

db = DBSCAN(eps=epsilon,
            min_samples=samples,
            metric='precomputed',
            algorithm='auto').fit(precomputed_mat)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print("\nUsing DBSCAN")
print("\nUsing eps: %0.3f" % (epsilon))
print("Using min_samples: %0.3f\n" % (samples))
print("Estimated number of clusters: %d\n" % n_clusters_)
print("Silhouette Coefficient: %0.3f" %
      metrics.silhouette_score(X, labels,
                               metric='cosine'))
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
print("Adjusted Rand Index: %0.3f" %
      metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f" %
      metrics.adjusted_mutual_info_score(labels_true, labels))
##############################################################################
# Validation - Reality check
#print y
print "\n Real clusters - labels validation, each line is a real cluster\n"
for item_index in xrange(0, len(db.labels_)):
    sys.stdout.write(" %s " % (db.labels_[item_index]))
    if (item_index == 23) or (item_index == 34) or (item_index == 44) or (item_index == 54) or \
    (item_index == 63) or (item_index == 70) or (item_index == 74):
        sys.stdout.write("\n")
print "\n\nEnd\n"
########################## LOG KEEPING #######################################
with open('log.txt', 'a') as file:
    file.write("\n##### DBSCAN CLUSTERING #####\n")
    file.write("Using eps: %0.3f \n" % (epsilon))
    file.write("Using min_samples: %0.3f \n" % (samples))
    file.write("\nHomogeneity: %0.3f\n" %
               metrics.homogeneity_score(y, db.labels_))
    file.write("V-measure: %0.3f\n" % metrics.v_measure_score(y, db.labels_))
    file.write("Adjusted Rand-Index: %.3f\n" %
               metrics.adjusted_rand_score(y, db.labels_))
    file.write("\nCluster Labels, each line is a real category\n")
    for item_index in xrange(0, len(db.labels_)):
        file.write(" %s " % (db.labels_[item_index]))
        if (item_index == 23) or (item_index == 34) or (item_index == 44) or (item_index == 54) or \
        (item_index == 63) or (item_index == 70) or (item_index == 74):
            file.write("\n")
    file.write("\n ### END OF CLUSTERING ### \n")
    file.close()
###############################################################################
