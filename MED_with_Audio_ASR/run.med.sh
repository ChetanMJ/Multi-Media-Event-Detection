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
feat_dim_mfcc=200
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py ${event} "kmeans/50_features.kmeans.npy" $feat_dim_mfcc mfcc_pred/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py mfcc_pred/svm.$event.model "kmeans/50_features.kmeans.npy" $feat_dim_mfcc mfcc_pred/${event}_mfcc.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label mfcc_pred/${event}_mfcc.lst
done

echo ""
echo "#####################################"
echo "#       MED with ASR Features       #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
feat_dim_asr=781
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "asrfeat/asr_tfidf_features.npy" $feat_dim_asr asr_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py asr_pred/svm.$event.model "asrfeat/asr_tfidf_features.npy" $feat_dim_asr asr_pred/${event}_asr.lst  ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label asr_pred/${event}_asr.lst
done



echo ""
echo "#####################################"
echo "#    MED with MFCC and ASR features #"
echo "#####################################"
mkdir -p asr_pred
# iterate over the events
feat_dim_asr=100
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py $event "mfcc_asr/all_video_features_mfcc_asr.npy" $feat_dim_asr mfcc_asr_pred/svm.$event.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py mfcc_asr_pred/svm.$event.model "mfcc_asr/all_video_features_mfcc_asr.npy" $feat_dim_asr mfcc_asr_pred/${event}_asr.lst  ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label mfcc_asr_pred/${event}_asr.lst
done
