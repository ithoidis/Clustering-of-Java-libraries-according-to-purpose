########################################################################
#A script used to seperate the data in arrays and do preprocessing to it
########################################################################
######Arrays created are:
#freq ---> frequency of every attribute in every instance
#attribute ---> name of every attribute (excluding project and category)
#project ---> the project in which every instance belongs to
#category ---> the category in which every instance belongs to

import numpy as np
import io
import datetime

#read the frequencies for each instance and keep it in a 2D array named freq
data = np.genfromtxt('../data/original/dataset.txt',delimiter=';')
freq = data[1:,2:]

# Delete last row of dataset containing garbage
freq = np.delete(freq, (-1), axis=0)
#print freq.shape

print "Number of instances:  %d" %(freq.shape[0])
print "Number of attributes: %d" %(freq.shape[1])

#read the name of each attribute in a 1D array (excluding project[0] and category[1])
names = open("../data/original/dataset.txt")
attribute = names.readline()
attribute = attribute.split(";")
attribute = attribute[2:]
print "Number of attribute names: %d" %(len(attribute))


#read project and category for every instance and save it in 2 1D arrays
# project: A vector containing the 80 names of the projects
# category: A lookup table containing the category of each project
project = np.empty([freq.shape[0]], dtype="S100")
category = np.empty([freq.shape[0]], dtype="S100")
for i in range(0,freq.shape[0]-1):
    temp = names.readline()
    temp = temp.split(";")
    project[i] = temp[0];
    category[i] = temp[1];
print "Number of projects:  %d" %(project.shape[0])

# Keep a log of the experiment
with open('log.txt', 'a') as file:
    file.write("\n\n#### Start of experiment #####\n")
    file.write(str(datetime.datetime.now())+"\n\n")
    file.write("Number of instances, projects:  %d \n" %(freq.shape[0]))
    file.write("Number of attributes: %d \n" %(freq.shape[1]))
    file.write("Number of attribute names: %d \n" %(len(attribute)))
    file.close()

#An example of how the dataset is stored and displayed
print "\nExample of how the dataset is stored"
print ""
print "project                           category        %s" %(attribute[0])
print "%s %s         %d" %(project[0], category[0], freq[0][0])
print "%s            %s         %d" %(project[1], category[1], freq[1][0])
print ""
print "#########################################################################"
