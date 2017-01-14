from preprocessDataset import freq, attribute, project, category, low_thresh, high_thresh, keepSize, norma
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
print precomputed_mat
plt.clf()
plt.imshow(precomputed_mat, vmin=0, vmax=1)
plt.savefig("low" + str(low_thresh) + "high" + str(high_thresh) + "keep" +
            str(keepSize) + "norma" + str(norma))
