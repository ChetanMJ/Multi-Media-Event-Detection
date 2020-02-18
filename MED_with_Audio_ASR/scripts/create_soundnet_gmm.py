#!/bin/python
import numpy
import os
import os.path
import pickle
from sklearn.cluster.k_means_ import KMeans
from sklearn.mixture import GaussianMixture
import sklearn.mixture._gaussian_mixture
import sklearn.mixture
import sys
import scipy
# Generate gmm features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} gmm_model, cluster_num, file_list".format(sys.argv[0]))
        print ("gmm_model -- path to the gmm model")
        print ("cluster_num -- number of cluster")
        print ("file_list -- the list of videos")
        exit(1)

    gmm_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the gmm model
    gmmx = pickle.load(open(gmm_model, 'rb'))
    
  
    files = open(file_list, "r")
    
    
    out_file = '/home/ubuntu/multimedia/11775-hws-master/hw1_code/soundnet16_gmm/50_features_soundnet16.gmm.npy'
    all_gmm = []
    
    for file in files:
    	file_name = '/home/ubuntu/multimedia/11775-hws-master/hw1_code/sound_net_16/'+file.strip()+ "_16.npy"
    	print(file_name)
    	if os.path.isfile(file_name):
    	        x1 = numpy.load(file_name)   	
		
    	        gmmo = gmmx.predict(x1)
    	
    	        hist, _ = numpy.histogram(gmmo, 50, density=False)
    	
    	        hist = hist/numpy.sum(hist)
    		   		
    	else:
    	        hist = numpy.zeros((50,))
    	
    	all_gmm.append(hist)
    	
    numpy.save(out_file, all_gmm)
    files.close()
    print ("gmm features generated successfully!")
