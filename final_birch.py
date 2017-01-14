import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist,pdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Birch
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import manifold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



#read the frequencies for each instance and keep it in a 2D array named freq
text = np.genfromtxt('C:\Users\ep\Desktop\Python\data.txt',delimiter=',')
data = text[1:,2:]


print "Number of instances:  %d" %(data.shape[0])
print "Number of attributes: %d" %(data.shape[1] )


#read the name of each attribute in a 1D array (excluding project and category)
names = open("C:\Users\ep\Desktop\Python\data.txt")
attribute = names.readline()
attribute = attribute.split(",")
attribute = attribute[2:]
print "Number of attribute names: %d" %(len(attribute))


#read project and category for every instance and save it in 2 1D arrays
project = np.empty([data.shape[0]], dtype="S100")
category = np.empty([data.shape[0]], dtype="S100")
for i in range(0,data.shape[0]):
    temp = names.readline()
    temp = temp.split(",")
    project[i] = temp[0];
    category[i] = temp[1];
print "Number of projects:  %d" %(project.shape[0])
print "Number of categories: %d" %(category.shape[0])
print "\n"


labels=category;

pipeline = Pipeline([('scaling', StandardScaler()), ('pca', PCA(n_components=2))])
X=pipeline.fit_transform(data)

def measures(estimator,data,labels):
	print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels, estimator.labels_)
	print "Completeness: %0.3f" % metrics.completeness_score(labels, estimator.labels_)
	print "V-measure: %0.3f" % metrics.v_measure_score(labels, estimator.labels_)
	print "Adjusted Rand-Index: %.3f" % metrics.adjusted_rand_score(labels, estimator.labels_)
	print "Silhouette Coefficient: %0.3f" % metrics.silhouette_score(data, estimator.labels_)
	centroids=estimator.subcluster_centers_
	euclid=cdist(data,centroids,'euclidean')
	dist=np.min(euclid,axis=1)
	wcss=sum(dist**2)#within cluster sum of squares
	print "Cohesion: %0.3f" % wcss 
	tss=sum(pdist(data)**2)/data.shape[0]#total sum of squares
	bss=tss-wcss #between cluster sum of squares
	print "Seperation: %0.3f" % bss
	print "\n"
	return wcss

#Birch hierarchical
bi1=Birch(threshold=0.003, n_clusters=None)
bi1.fit(X)
y_pred= bi1.labels_.astype(np.int)
fig9 = plt.figure(9)
plt.title('threshold=0.003, n_clusters=None')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
wcss_1 = measures(bi1,X,labels)
np.savetxt('newData_bi1.txt',bi1.labels_,fmt="%s",delimiter=",")


bi2=Birch(threshold=0.016, n_clusters=None)
bi2.fit(X)
y_pred= bi2.labels_.astype(np.int)
fig10 = plt.figure(10)
plt.title('threshold=0.016, n_clusters=None')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
wcss_2 = measures(bi2,X,labels)
np.savetxt('newData_bi2.txt',bi2.labels_,fmt="%s",delimiter=",")

bi3=Birch(threshold=0.016, n_clusters = 8)
bi3.fit(X)
y_pred= bi1.labels_.astype(np.int)
fig11 = plt.figure(11)
plt.title('threshold=0.016, n_clusters = 8')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
wcss_3 = measures(bi3,X,labels)
np.savetxt('newData_bi3.txt',bi3.labels_,fmt="%s",delimiter=",")

bi4=Birch(threshold=0.016, branching_factor=75)
bi4.fit(X)
y_pred= bi1.labels_.astype(np.int)
fig12 = plt.figure(12)
plt.title('threshold=0.016, n_clusters=None branching_factor=75')
plt.scatter(X[:,0], X[:,1], c = y_pred, cmap = plt.cm.Paired)
plt.show()
wcss_4 = measures(bi4,X,labels)
np.savetxt('newData_bi4.txt',bi4.labels_,fmt="%s",delimiter=",")


np.savetxt('wcsses.txt',[wcss_1, wcss_2, wcss_3, wcss_4],fmt="%s",delimiter=",")

