# DONT FORGET TO USE THE -B ARGUMENT TO SUPPRESS THE .pyc FILE CREATION
#
# low_thresh and high_thresh are thresholds for the minimum and maximum Number
# of instances in which a certain attribute is present. The attributes outside of
# those thresholds are kept in a list named removeList.
#
# row-wise normalization is performed on the resulting array
# the best keepSize attributes are kept for every project

from readDataset import freq, attribute, project, category
import numpy as np
import scipy as sc
import sys
import sklearn.preprocessing as pr

# freq.shape[0] is the number of the instances
# freq.shape[1] is the number of the attributes
low_thresh = int(sys.argv[1])
high_thresh = int(sys.argv[2])
keepSize = int(sys.argv[3])
norma = int(sys.argv[4])
removeList = []

print "\nLow Threshold: %d" %(low_thresh)
print "High Threshold: %d" %(high_thresh)
print "Keep Size: %d" %(keepSize)
 
## Counts in how many projects a word is appeared, then removes words based on the thresholds
# For example if low_tresh is 3, words tha appear only in 1 or 2 projects are kicked out
# The index of the attribute(word) to be removed is hold to removeList 
# for each attribute(word) i
for i in range(0,freq.shape[1]):
    count = 0
    for j in range(0,freq.shape[0]):  # for each instance(project)
        if (freq[j][i] != 0):  # if the word appears in this project we raise the counter
            count = count + 1
    if ((count > high_thresh) or (count < low_thresh)):
        removeList.append(i)

print "Number of attributes: %d" %(freq.shape[1])
#print "Removable attributes: %d" %(len(removeList))
#print "Attributes remaining: %d \n" %(freq.shape[1] - len(removeList))

#print "'freq' array dimensions before delete" + str(freq.shape)
#print "'attribute' array dimensions before delete" + str(attribute.shape)
# Delete the attributes in removeList from freq
freq = sc.delete(freq, removeList, 1)
attribute = sc.delete(attribute, removeList, 0)
#print "Number of attributes after first cleanup: %d" %(freq.shape[1])
#print "'freq' array dimensions after delete" + str(freq.shape)
#print "'attribute' array dimensions after delete" + str(attribute.shape)


# Keep log
with open('log.txt', 'a') as file:
    file.write("\nRemovable attributes after first cleanup: %d \n" %(len(removeList)))
    file.write("Attributes remaining after first cleanup: %d \n" %(freq.shape[1]))
    file.close()    

# Normalize row-wise, the number of times a word is appeared in a project is divided by all the words appeared in a project
#row_sums = freq.sum(axis=1)  # row_sums is a list containing the number of words in each project
#freq = freq / row_sums[:, np.newaxis]
 
# L2 normalisation for each instance, instead of paragka kanonikopoihsh
if(norma==1):
    freq = pr.normalize(freq, norm='l2', axis=1)

#Keep the best keepSize attributes for each project.
topAttributes = np.empty([freq.shape[0],keepSize], dtype=int)
for j in range(0,freq.shape[0]):
    currentProject = freq[j,:]
    sortedFrequencies = np.argsort(currentProject)
    topAttributes[j][:] = sortedFrequencies[-keepSize:]
keepAtr = np.reshape(topAttributes,[freq.shape[0]*keepSize])
keepAtr = np.unique(keepAtr)
keepAtr = np.sort(keepAtr)
attribute = attribute[keepAtr]
freq = freq[:,keepAtr]

print "Number of attributes after cleanup: %d \n" %(freq.shape[1])
#print "The list with the remaining attributes is: \n"
#print attribute

# Keep log
with open('log.txt', 'a') as file:
    file.write('Attributes remaining after second cleanup: %d\n\n' %(freq.shape[1]))
    display_counter = 0
    for item in attribute:
        file.write(" %s "%item)
        display_counter = display_counter + 1
	if (display_counter == 10):
		file.write("\n")
		display_counter = 0;
    file.write('\n\nUsing keepSize = %d\n' %(keepSize))
    file.write('low_thresh = %d\n' %(low_thresh))
    file.write('high_tresh = %d\n' %(high_thresh))
    file.write("\n #### END OF DATA PREPROCESSING #### ")
    file.close()
