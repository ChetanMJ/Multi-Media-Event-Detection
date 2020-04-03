#!/bin/python 

import numpy as np


Mobilenetv2_features = np.load('mobilenetv2_1000_Imgnet_features.npy')
SoundNet16_kmeans_features = np.load('50_features_soundnet16.kmeans.npy')
ASR_features = np.load('asr_tfidf_features.npy')
SoundNet16_gmm_features = np.load('50_features_soundnet16.gmm.npy')


vedio_count = Mobilenetv2_features.shape[0]


### early fusion 1 : MobilenetV2 + soundnet16_kmeans + ASR

EF_MNetV2_SN16kmeans_ASR_features = []

for i in range(vedio_count):
    MV2 = Mobilenetv2_features[i]
    SN16Kmeans = SoundNet16_kmeans_features[i]
    ASR = ASR_features[i]
    EF = np.append(MV2, SN16Kmeans)
    EF = np.append(EF, ASR)
    EF_MNetV2_SN16kmeans_ASR_features.append(EF)

EF_MNetV2_SN16kmeans_ASR_features = np.array(EF_MNetV2_SN16kmeans_ASR_features)


np.save('EF_MNetV2_SN16kmeans_ASR_features.npy', EF_MNetV2_SN16kmeans_ASR_features)


### early fusion 2 : MobilenetV2 + soundnet16_gmm + ASR

EF_MNetV2_SN16gmm_ASR_features = []

for i in range(vedio_count):
    MV2 = Mobilenetv2_features[i]
    SN16gmm = SoundNet16_gmm_features[i]
    ASR = ASR_features[i]
    EF = np.append(MV2, SN16gmm)
    EF = np.append(EF, ASR)
    EF_MNetV2_SN16gmm_ASR_features.append(EF)

EF_MNetV2_SN16gmm_ASR_features = np.array(EF_MNetV2_SN16gmm_ASR_features)
np.save('EF_MNetV2_SN16gmm_ASR_features.npy', EF_MNetV2_SN16gmm_ASR_features)


### early fusion 3 : MobilenetV2 + soundnet16_gmm

EF_MNetV2_SN16gmm_features = []

for i in range(vedio_count):
    MV2 = Mobilenetv2_features[i]
    SN16gmm = SoundNet16_gmm_features[i]
    EF = np.append(MV2, SN16gmm)
    EF_MNetV2_SN16gmm_features.append(EF)

EF_MNetV2_SN16gmm_features = np.array(EF_MNetV2_SN16gmm_features)
np.save('EF_MNetV2_SN16gmm_features.npy', EF_MNetV2_SN16gmm_features)

### early fusion 4 : MobilenetV2 + ASR

EF_MNetV2_ASR_features = []

for i in range(vedio_count):
    MV2 = Mobilenetv2_features[i]
    ASR = ASR_features[i]
    EF = np.append(MV2, ASR)
    EF_MNetV2_ASR_features.append(EF)

EF_MNetV2_ASR_features = np.array(EF_MNetV2_ASR_features)
np.save('EF_MNetV2_ASR_features.npy', EF_MNetV2_ASR_features)




