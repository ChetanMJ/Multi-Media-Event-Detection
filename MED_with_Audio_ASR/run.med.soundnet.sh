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
  python scripts/train_svm.py ${event} "sound_net_10/50_features_soundnet.kmeans.npy" $feat_dim_mfcc sound_net_10/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py sound_net_10/svm.${event}.model "sound_net_10/50_features_soundnet.kmeans.npy" $feat_dim_mfcc sound_net_10/${event}_soundnet.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label sound_net_10/${event}_soundnet.lst
done



feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py ${event} "sound_net_16/50_features_soundnet16.kmeans.npy" $feat_dim_mfcc sound_net_16/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py sound_net_16/svm.${event}.model "sound_net_16/50_features_soundnet16.kmeans.npy" $feat_dim_mfcc sound_net_16/${event}_soundnet.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label sound_net_16/${event}_soundnet.lst
done




feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py ${event} "soundnet16_asr/all_video_features_sound_asr2.npy" $feat_dim_mfcc soundnet16_asr/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  python scripts/test_svm.py soundnet16_asr/svm.${event}.model "soundnet16_asr/all_video_features_sound_asr2.npy" $feat_dim_mfcc soundnet16_asr/${event}_soundnet.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label soundnet16_asr/${event}_soundnet.lst
done




feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py ${event} "soundnet16_gmm/50_features_soundnet16.gmm.npy" $feat_dim_mfcc soundnet16_gmm/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python scripts/test_svm.py soundnet16_gmm/svm.${event}.model "soundnet16_gmm/50_features_soundnet16.gmm.npy" $feat_dim_mfcc soundnet16_gmm/${event}_soundnet.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label soundnet16_gmm/${event}_soundnet.lst
done



feat_dim_mfcc=50
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  python scripts/train_svm.py ${event} "soundnet16_gmm_asr/all_video_features_sound_gmm_asr.npy" $feat_dim_mfcc soundnet16_gmm_asr/svm.${event}.model || exit 1;
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred
  python scripts/test_svm.py soundnet16_gmm_asr/svm.${event}.model "soundnet16_gmm_asr/all_video_features_sound_gmm_asr.npy" $feat_dim_mfcc soundnet16_gmm_asr/${event}_soundnet.lst ${event}|| exit 1;
  # compute the average precision by calling the mAP package
  ap list/${event}_val_label soundnet16_gmm_asr/${event}_soundnet.lst
done


