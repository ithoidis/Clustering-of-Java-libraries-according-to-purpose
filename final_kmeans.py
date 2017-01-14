import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist
from mpl_toolkits.mplot3d import Axes3D
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
#data=normalize(data)
#np.savetxt('target.txt',labels,fmt="%s",delimiter=",")

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
X=pipeline.fit_transform(data)

#X=manifold.Isomap(3,2).fit_transform(data)
def measures(estimator,data,labels):
    print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, estimator.labels_)
    print "Completeness: %0.3f" % metrics.completeness_score(labels, estimator.labels_)
    print "V-measure: %0.3f" % metrics.v_measure_score(labels, estimator.labels_)
    print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, estimator.labels_)
    print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, estimator.labels_)
    centroids=estimator.cluster_centers_
    euclid=cdist(data,centroids,'euclidean')
    dist=np.min(euclid,axis=1)
    wcss=sum(dist**2)#within cluster sum of squares
    print "Cohesion: %0.3f" % wcss 
    tss=sum(pdist(data)**2)/data.shape[0]#total sum of squares
    bss=tss-wcss #between cluster sum of squares
    print "Seperation: %0.3f" % bss
    return dist




#KMeans centroid based
km1=KMeans(n_clusters=8,init='k-means++',n_init=10) #default parameters
km1.fit(X)
y_pred= km1.labels_.astype(np.int)
fig1 = plt.figure(1)
plt.title('k-Means (init = k-means++, n_init = 10)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
d1=measures(km1,X,labels)

km2=KMeans(n_clusters=18,init='k-means++',n_init=10)
km2.fit(X)
y_pred= km2.labels_.astype(np.int)
fig2 = plt.figure(2)
plt.title('k-Means (init = random, n_init = 10)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
d2=measures(km2,X,labels)

km3=KMeans(n_clusters=28,init='k-means++',n_init=10)
km3.fit(X)
y_pred= km3.labels_.astype(np.int)
fig3 = plt.figure(3)
plt.title('k-Means (init = k-means++, n_init = 7)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
d3=measures(km3,X,labels)

km4=KMeans(n_clusters=38,init='random',n_init=10)
km4.fit(X)
y_pred= km4.labels_.astype(np.int)
fig4 = plt.figure(4)
plt.title('k-Means (init = random, n_init = 7)')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
d4=measures(km4,X,labels)



