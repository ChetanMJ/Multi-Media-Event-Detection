#!/bin/python
import numpy
import numpy as np
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

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = numpy.load(kmeans_model)
    
  
    files = open(file_list, "r")

    out_file = 'kmeans/50_features_surf.kmeans.npy'
    all_kmeans = []
    
    for file in files:
    	video_kmeans = []
    	for i in range(6):
            file_name = 'surf/'+file.strip()+ '.surf_'+str(i)+'.npy'
            if os.path.isfile(file_name):
                x1 = numpy.load(file_name)
                euclead_dist = scipy.spatial.distance.cdist(x1, kmeans, 'euclidean')
                argmin_euclead_dist = numpy.argmin(euclead_dist, axis = 1)
                hist, _ = numpy.histogram(argmin_euclead_dist, 50, density=False)
                hist = hist/numpy.sum(hist)
            else:
                hist = numpy.zeros((50,))

            video_kmeans.append(hist)

    	video_kmeans_aveg = np.max(np.array(video_kmeans), axis=0)
    	print(video_kmeans_aveg.shape)
    	all_kmeans.append(video_kmeans_aveg)
    	
    all_kmeans = np.array(all_kmeans)
    numpy.save(out_file, all_kmeans)

    files.close()	

    print ("K-means features generated successfully!")
