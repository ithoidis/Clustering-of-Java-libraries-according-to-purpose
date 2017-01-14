import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics
from preprocessDataset import freq, attribute, project, category
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy as sc
import sys

X, y = freq, category

est = KMeans(n_clusters=8,
             init='k-means++',
             n_init=10,
             max_iter=3000,
             tol=0.000001,
             precompute_distances='auto',
             verbose=0,
             random_state=None,
             copy_x=True,
             n_jobs=4)
est.fit(X, y)
#print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(
#    X, est.labels_, metric='l2',
#    sample_size=None)
#for j in range(0,10):
#    est = KMeans(n_clusters=8, init=est.cluster_centers_, n_init=1, max_iter=3000, tol=0.000001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=4)
#    est.fit(X,y)
print("\nUsing k-means\n")

print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, est.labels_, metric='cosine')
#silCoeff = metrics.silhouette_score(X, est.labels_,sample_size=None)
#homCoeff = metrics.homogeneity_score(y, est.labels_)

#import matplotlib.pyplot as plt
#plt.plot(range(0,numClusters), silCoeff, 'ro')
#plt.axis([0, numClusters, 0, 1])
#plt.show()

print "Homogeneity: %0.3f" % metrics.homogeneity_score(y, est.labels_)
print "Completeness: %0.3f" % metrics.completeness_score(y, est.labels_)
print "V-measure: %0.3f" % metrics.v_measure_score(y, est.labels_)
print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(y, est.labels_)
#print len(metrics.silhouette_samples(X, est.labels_))
#print "Cohesion - Within cluster SSE: %0.3f" % (est.inertia_)

############### Validation - Reality check ###########
#print y
print "\n Real clusters - labels validation, each line is a real cluster\n"
for item_index in xrange(0, len(est.labels_)):
    sys.stdout.write(" %s " % (est.labels_[item_index]))
    if (item_index == 23) or (item_index == 34) or (item_index == 44) or (item_index == 54) or \
    (item_index == 63) or (item_index == 70) or (item_index == 74):
        sys.stdout.write("\n")
print "\n\nEnd\n"

################## LOG KEEPING ########################
with open('log.txt', 'a') as file:
    file.write("\n##### KMEANS CLUSTERING #####")
    file.write("\nHomogeneity: %0.3f\n" %
               metrics.homogeneity_score(y, est.labels_))
    file.write("V-measure: %0.3f\n" % metrics.v_measure_score(y, est.labels_))
    file.write("Adjusted Rand-Index: %.3f\n" %
               metrics.adjusted_rand_score(y, est.labels_))
    file.write("Cohesion - Within cluster SSE: %0.3f\n" % (est.inertia_))
    file.write("\nCluster Labels, each line is a real category\n")
    for item_index in xrange(0, len(est.labels_)):
        file.write(" %s " % (est.labels_[item_index]))
        if (item_index == 23) or (item_index == 34) or (item_index == 44) or (item_index == 54) or \
        (item_index == 63) or (item_index == 70) or (item_index == 74):
            file.write("\n")
    file.write("\n ### END OF CLUSTERING ### \n")
    file.close()
#########################################################
