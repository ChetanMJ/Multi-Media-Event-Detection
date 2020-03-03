#!/bin/bash

# An example script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Note that this script gives you the very basic setup. Its configuration is by no means the optimal. 
# This is NOT the only solution by which you approach the problem. We highly encourage you to create
# your own setups. 

# Paths to different tools; 
map_path=/home/ubuntu/multimedia/11775-hws-master/mAP
export PATH=$map_path:$PATH

echo "#####################################"
echo "#       MED with MFCC Features      #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_svm.py ${event} "kmeans/50_features_surf.kmeans.npy" $feat_dim_mfcc kmeans/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python test_svm.py kmeans/svm.${event}.model "kmeans/50_features_surf.kmeans.npy" $feat_dim_mfcc kmeans/${event}_surf.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label kmeans/${event}_surf.lst
done

echo "#####################################"
echo "#       MED with CNN 1000 Features    #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_svm.py ${event} "mobilenetv2_1000_Imgnet_features.npy" $feat_dim_mfcc cnn/cnn1000.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python test_svm.py cnn/cnn1000.${event}.model "mobilenetv2_1000_Imgnet_features.npy" $feat_dim_mfcc cnn/${event}_cnn1000.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label cnn/${event}_cnn1000.lst
done



echo "#####################################"
echo "#       MED with CNN 34 Features    #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_svm.py ${event} "mobilenetv2_features.npy" $feat_dim_mfcc cnn/cnn34.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python test_svm.py cnn/cnn34.${event}.model "mobilenetv2_features.npy" $feat_dim_mfcc cnn/${event}_cnn34.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label cnn/${event}_cnn34.lst
done




echo "#####################################"
echo "#       MED with CNN 34 Features with Kfold    #"
echo "#####################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_kfold_svm.py ${event} "mobilenetv2_features.npy" $feat_dim_mfcc cnn/cnn34_kfold.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python test_svm.py cnn/cnn34_kfold.${event}.model "mobilenetv2_features.npy" $feat_dim_mfcc cnn/${event}_cnn34_kfold.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label cnn/${event}_cnn34_kfold.lst
done




echo "##################################################"
echo "#       MED with CNN 1000 Features with Kfold    #"
echo "##################################################"
mkdir -p mfcc_pred
# iterate over the events
feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python train_kfold_svm.py ${event} "mobilenetv2_1000_Imgnet_features.npy" $feat_dim_mfcc cnn/cnn1000_kfold.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python test_svm.py cnn/cnn1000_kfold.${event}.model "mobilenetv2_1000_Imgnet_features.npy" $feat_dim_mfcc cnn/${event}_cnn1000_kfold.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label cnn/${event}_cnn1000_kfold.lst
done


