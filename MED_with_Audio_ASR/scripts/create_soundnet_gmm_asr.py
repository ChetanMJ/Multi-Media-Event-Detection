
#################### reduce 783 dimensions in asr to 50 #########################
from sklearn.manifold import TSNE

import numpy as np
all_video_features = np.load('/home/ubuntu/multimedia/11775-hws-master/hw1_code/asrfeat/asr_tfidf_features.npy')

#tsne = TSNE(n_components=20, perplexity=10, method='exact')
#asr_tfidf_features_normalized_tsne2d = tsne.fit_transform(all_video_features)

#np.save('/home/ubuntu/multimedia/11775-hws-master/hw1_code/soundnet16_asr/asr_tfidf_features_normalized_tsne2d', asr_tfidf_features_normalized_tsne2d)

#################################################################################


#################### combine features from soundnet16 and asr normalized ###############
val_video = 'list/val'
all_videos = 'list/all.video'

#all_video_features_asr = np.load('/home/ubuntu/multimedia/11775-hws-master/hw1_code/soundnet16_asr/asr_tfidf_features_normalized_tsne2d.npy')
all_video_features_sound = np.load('/home/ubuntu/multimedia/11775-hws-master/hw1_code/soundnet16_gmm/50_features_soundnet16.gmm.npy')
all_video_features_asr=all_video_features


all_video_features_soundnet_asr=[]
for i in range(all_video_features_asr.shape[0]):
    fetaure_sound = all_video_features_sound[i]
    fetaure_asr = all_video_features_asr[i]
    fetaure_sound_asr = np.append(fetaure_sound, fetaure_asr)
    all_video_features_soundnet_asr.append(fetaure_sound_asr)

all_video_features_soundnet_asr = np.array(all_video_features_soundnet_asr)    
np.save('/home/ubuntu/multimedia/11775-hws-master/hw1_code/soundnet16_gmm_asr/all_video_features_sound_gmm_asr.npy', all_video_features_soundnet_asr)

###################################################################################
