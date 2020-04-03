#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import pickle
import sys
import numpy as np

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print ("model_file -- path of the trained svm file")
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features; provided just for debugging")
        print ("output_file -- path to save the prediction score")

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    event_name = sys.argv[5]


    ##############load best model#########################################
    model = pickle.load(open(model_file, 'rb'))
    ######################################################################
    
    
    ### laod all file features and labels ################################
    val_video = 'list/val'
    all_videos = 'list/all.video'
    test_videos = 'list/test.video'
    
    all_video_features = np.load(feat_dir)
    
    all_videos_list=[]
    files = open(all_videos, "r")
    for file in files:
        all_videos_list.append(file.strip())
    files.close()
    
    ######################################################################
    
    
    ################## extract test file features#########################
    test_videos_list=[]
    test_feature_list = []
    
    event=event_name
    
    files = open(test_videos, "r")
    for file in files:
        test_videos_list.append(file.strip())
        vedio_index = all_videos_list.index(file.strip())
        test_feature_list.append(all_video_features[vedio_index]) 
            
    files.close()
   
    
    test_feature_list = np.array(test_feature_list)
    ########################################################################
    
    
    
    ################## extract validation file features and labels########
    val_videos_list=[]
    val_label_list=[]
    val_feature_list = []
    
    event=event_name
    
    files = open(val_video, "r")
    label_list_file = open('list/'+event+'_val_label', "w")
    for file in files:
        val_videos_list.append(file.strip().split()[0])
        label = file.strip().split()[1]
        if label == event:
            label_list_file.write(str(1)+'\n')
        else:
            label_list_file.write(str(0)+'\n')
            
        vedio_index = all_videos_list.index(file.strip().split()[0])
        val_feature_list.append(all_video_features[vedio_index]) 
            
    files.close()
    label_list_file.close()
    
    
    val_feature_list = np.array(val_feature_list)
    ########################################################################    
    
    
    
    
    ################### Predict scores for Validation features##############
    val_predict_scores = model.predict_proba(val_feature_list)
    test_predict_scores = model.predict_proba(test_feature_list)
    
    ################### writes predictions to a file #######################
    label_list_file = open(output_file, "w")
    
    for scores in val_predict_scores:
        label_list_file.write(str(scores[1])+'\n')
        
    label_list_file.close()

    label_list_file = open(output_file+".test", "w")
    
    for scores in test_predict_scores:
        label_list_file.write(str(scores[1])+'\n')
        
    label_list_file.close()
    
    #########################################################################