#!/bin/python
import numpy
import os
import os.path
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys
import scipy
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print ("kmeans_model -- path to the kmeans model")
        print ("cluster_num -- number of cluster")
        print ("file_list -- the list of videos")
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = numpy.load(kmeans_model)
    
  
    files = open(file_list, "r")
    
    
    out_file = './sound_net_16/50_features_soundnet16.kmeans.npy'
    all_kmeans = []
    
    for file in files:
    
    	file_name = './sound_net_16/'+file.strip()+ "_16.npy"
    	print(file_name)
    	if os.path.isfile(file_name):
    	        x1 = numpy.load(file_name)   	
    	        euclead_dist = scipy.spatial.distance.cdist(x1, kmeans, 'euclidean')
	
    	        argmin_euclead_dist = numpy.argmin(euclead_dist, axis = 1)
    	
    	        hist, _ = numpy.histogram(argmin_euclead_dist, 50, density=False)
    	
    	        hist = hist/numpy.sum(hist)
    		   		
    	else:
    	        hist = numpy.zeros((50,))
    	
    	all_kmeans.append(hist)
    	
    numpy.save(out_file, all_kmeans)

    files.close()	
    

    print ("K-means features generated successfully!")
