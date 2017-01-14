import numpy as np
from sklearn.cluster import Birch
from preprocessDataset import freq, attribute, project, category
from sklearn import metrics

from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import sys
X, y = freq, category
labels_true = y

branch_fact = 50
clusters = 8
thresh = 0.60

brc = Birch(branching_factor=branch_fact,
            n_clusters=clusters,
            threshold=thresh,
            compute_labels=True)
brc.fit(X)
brc.predict(X)
labels = brc.labels_
print("\nUsing BIRCH")
print("\nUsing branching_factor: %0.3f" % (branch_fact))
print("Using n_clusters: %0.3f" % (clusters))
print("Using threshold: %0.3f \n" % (thresh))
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

#print("Precom Silhouette Coefficient: %0.3f" % metrics.silhouette_score(precomputed_mat, labels, metric='precomputed'))


##############################################################################
# Validation - Reality check
#print y
print "\n Real clusters - labels validation, each line is a real cluster\n"
for item_index in xrange(0, len(brc.labels_)):
    sys.stdout.write(" %s " % (brc.labels_[item_index]))
    if (item_index == 23) or (item_index == 34) or (item_index == 44) or (item_index == 54) or \
    (item_index == 63) or (item_index == 70) or (item_index == 74):
        sys.stdout.write("\n")
print "\n\nEnd\n"
########################## LOG KEEPING #######################################
with open('log.txt', 'a') as file:
    file.write("\n##### BIRCH CLUSTERING #####\n")
    file.write("Using branching_factor: %0.3f \n" % (branch_fact))
    file.write("Using n_clusters: %0.3f \n" % (clusters))
    file.write("Using threshold: %0.3f \n" % (thresh))
    file.write("\nHomogeneity: %0.3f\n" %
               metrics.homogeneity_score(y, brc.labels_))
    file.write("V-measure: %0.3f\n" % metrics.v_measure_score(y, brc.labels_))
    file.write("Adjusted Rand-Index: %.3f\n" %
               metrics.adjusted_rand_score(y, brc.labels_))
    file.write("\nCluster Labels, each line is a real category\n")
    for item_index in xrange(0, len(brc.labels_)):
        file.write(" %s " % (brc.labels_[item_index]))
        if (item_index == 23) or (item_index == 34) or (item_index == 44) or (item_index == 54) or \
        (item_index == 63) or (item_index == 70) or (item_index == 74):
            file.write("\n")
    file.write("\n ### END OF CLUSTERING ### \n")
    file.close()
###############################################################################
