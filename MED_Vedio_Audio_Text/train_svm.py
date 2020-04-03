#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import pickle
import sys
import numpy as np
from sklearn import svm
import sklearn.metrics

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print ("Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0]))
        print ("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print ("feat_dir -- dir of feature files")
        print ("feat_dim -- dim of features")
        print ("output_file -- path to save the svm model")
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    
    
    ################ Regularise train data ####################################
    
    event=event_name
    
    
    all_videos = 'list/all.video'
    train_video = 'list/train'
    val_video = 'list/val'
    
   
    all_videos_list=[]
    files = open(all_videos, "r")
    for file in files:
        all_videos_list.append(file.strip())
    files.close()
    
    #all_video_features = np.load('./asrfeat/asr_tfidf_features.npy')
    all_video_features = np.load(feat_dir)
    
    train_videos_list=[]
    train_label_list=[]
    train_feature_list = []
    
    files = open(train_video, "r")
    P001_CNT = 0
    P002_CNT = 0
    P003_CNT = 0
    NULL_CNT = 0
    
    for file in files:
        label = file.strip().split()[1]
        
        vedio_index = all_videos_list.index(file.strip().split()[0])    
        
        if label == event:
            train_label_list.append(label)
            train_feature_list.append(all_video_features[vedio_index])
            train_videos_list.append(file.strip().split()[0])
      
        elif label == 'NULL':
            if NULL_CNT < 35:
                train_label_list.append('NOT_'+event)
                train_feature_list.append(all_video_features[vedio_index])
                train_videos_list.append(file.strip().split()[0])
                NULL_CNT = NULL_CNT + 1    
        else:
            if label == 'P001' and P001_CNT<=18:
                P001_CNT = P001_CNT+1
                train_label_list.append('NOT_'+event)
                train_feature_list.append(all_video_features[vedio_index])
                train_videos_list.append(file.strip().split()[0])
                
            if label == 'P002' and P002_CNT<=18:
                P002_CNT = P002_CNT+1
                train_label_list.append('NOT_'+event)
                train_feature_list.append(all_video_features[vedio_index])
                train_videos_list.append(file.strip().split()[0])
                
            if label == 'P003' and P003_CNT<=18:
                P003_CNT = P003_CNT+1
                train_label_list.append('NOT_'+event)
                train_feature_list.append(all_video_features[vedio_index])
                train_videos_list.append(file.strip().split()[0])
    
            
    files.close()
    train_feature_list = np.array(train_feature_list)
    train_label_list = np.array(train_label_list)
    
    val_videos_list=[]
    val_label_list=[]
    val_feature_list = []
    val_label_list_num=[]
    
    files = open(val_video, "r")
    for file in files:
        val_videos_list.append(file.strip().split()[0])
        label = file.strip().split()[1]
        if label == event:
            val_label_list.append(label)
            val_label_list_num.append(1)
        else:
            val_label_list.append('NOT_'+event)
            val_label_list_num.append(0)
            
        vedio_index = all_videos_list.index(file.strip().split()[0])
        val_feature_list.append(all_video_features[vedio_index])        
    files.close()
    
    val_feature_list = np.array(val_feature_list)
    val_label_list = np.array(val_label_list)
    val_label_list_num=np.array(val_label_list_num)

    
    ####################################################################################
    
    
    ################### Train and select best model ####################################
    
    kernel_type = ['linear', 'poly', 'rbf', 'sigmoid']
    regparam = [0.01, 0.03, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0, 100.0, 110.0]
    gamma_type = ['scale', 'auto']

    from sklearn import svm

    best_ap = None
    best_kernel = None
    best_regularization_param = None
    best_model = None
    best_gamma = None


    for kt in kernel_type:
        for rp in regparam:
            for gt in gamma_type:
            	SVM_classifier = svm.SVC(C=rp,  kernel=kt, gamma=gt, probability=True)
            	SVM_classifier.fit(train_feature_list, train_label_list)
            	predict = SVM_classifier.predict_proba(val_feature_list)
        
            	prediction_score = predict[:,1]
            	average_precision = sklearn.metrics.average_precision_score(val_label_list_num, prediction_score)
        
            	if (best_ap == None) or (average_precision > best_ap):
            	   best_ap = average_precision
            	   best_kernel = kt
            	   best_regularization_param = rp
            	   best_model = SVM_classifier
            	   best_gamma = gt
               
               

    #################################################################################
    
    ########### save best model and it associated params ######################
    

    pickle.dump(best_model, open(output_file, 'wb'))
    

    param_file = open(output_file+'.params', "w")
    param_file.write("Average precision: "+str(best_ap))
    param_file.write("\nKernel: "+(best_kernel))
    param_file.write("\nRegularization Param(C): "+str(best_regularization_param))
    param_file.write("\nGamma: "+(best_gamma))
    param_file.close()
    
    


    print ('SVM trained successfully for event %s!' % (event_name))
