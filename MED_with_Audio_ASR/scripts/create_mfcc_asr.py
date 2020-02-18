
#################### reduce 783 dimensions in asr to 50 #########################
from sklearn.manifold import TSNE

import numpy as np
all_video_features = np.load('/home/ubuntu/multimedia/11775-hws-master/hw1_code/asrfeat/asr_tfidf_features.npy')

#tsne = TSNE(n_components=50, perplexity=10, method='exact')
#asr_tfidf_features_normalized_tsne2d = tsne.fit_transform(all_video_features)
#np.save('asr_tfidf_features_normalized_tsne2d', asr_tfidf_features_normalized_tsne2d)

#################################################################################


#################### combine features from mfcc and asr normalized ###############
val_video = 'list/val'
all_videos = 'list/all.video'

#all_video_features_asr = np.load('./mfcc_asr/asr_tfidf_features_normalized_tsne2d.npy')
all_video_features_mfcc = np.load('/home/ubuntu/multimedia/11775-hws-master/hw1_code/kmeans/50_features.kmeans.npy')
all_video_features_asr = all_video_features

all_video_features_mfcc_asr=[]
for i in range(all_video_features_asr.shape[0]):
    fetaure_mfcc = all_video_features_mfcc[i]
    fetaure_asr = all_video_features_asr[i]
    fetaure_mfcc_asr = np.append(fetaure_mfcc, fetaure_asr)
    all_video_features_mfcc_asr.append(fetaure_mfcc_asr)

all_video_features_mfcc_asr = np.array(all_video_features_mfcc_asr)    
np.save('/home/ubuntu/multimedia/11775-hws-master/hw1_code/mfcc_asr/all_video_features_mfcc_asr2.npy', all_video_features_mfcc_asr)

###################################################################################