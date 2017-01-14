import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist,pdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale,Normalizer,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn import manifold
from mpl_toolkits.mplot3d import Axes3D


#read the frequencies for each instance and keep it in a 2D array named freq
text = np.genfromtxt('C:\Users\stergios\Desktop\data.txt',delimiter=',')
data = text[1:,2:]
#print "Number of instances:  %d" %(data.shape[0])
#print "Number of attributes: %d" %(data.shape[1] )


#read the name of each attribute in a 1D array (excluding project and category)
names = open("C:\Users\stergios\Desktop\data.txt")
attribute = names.readline()
attribute = attribute.split(",")
attribute = attribute[2:]
#print "Number of attribute names: %d" %(len(attribute))


#read project and category for every instance and save it in 2 1D arrays
project = np.empty([data.shape[0]], dtype="S100")
category = np.empty([data.shape[0]], dtype="S100")
for i in range(0,data.shape[0]):
    temp = names.readline()
    temp = temp.split(",")
    project[i] = temp[0];
    category[i] = temp[1];
#print "Number of projects:  %d" %(project.shape[0])
#print "Number of categories: %d" %(category.shape[0])


labels=category;

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
X=pipeline.fit_transform(data)


def measures(estimator,data,labels):
    n_clusters_ = len(set(estimator.labels_)) - (1 if -1 in estimator.labels_ else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, estimator.labels_)
    print "Completeness: %0.3f" % metrics.completeness_score(labels, estimator.labels_)
    print "V-measure: %0.3f" % metrics.v_measure_score(labels, estimator.labels_)
    print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, estimator.labels_)
    print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, estimator.labels_)
	

###DBSCAN density based 
db1=DBSCAN(eps=5, min_samples=1,algorithm='auto')
db1.fit(X)
y_pred= db1.labels_.astype(np.int)
fig5 = plt.figure(5)
plt.title('DBSCAN (algorithm = auto)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
measures(db1,X,labels)

db2=DBSCAN(eps=5, min_samples=1,algorithm='ball_tree')
db2.fit(X)
y_pred= db2.labels_.astype(np.int)
fig6 = plt.figure(6)
plt.title('DBSCAN (algorithm = ball_tree)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
measures(db2,X,labels)

db3=DBSCAN(eps=5, min_samples=1,algorithm='kd_tree')
db3.fit(X)
y_pred= db3.labels_.astype(np.int)
fig77 = plt.figure(7)
plt.title('DBSCAN (algorithm = kd_tree)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
measures(db3,X,labels)

db4=DBSCAN(eps=5, min_samples=1,algorithm='brute')
db4.fit(X)
y_pred= db4.labels_.astype(np.int)
fig8 = plt.figure(8)
plt.title('DBSCAN (algorithm = brute)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
measures(db4,X,labels)




