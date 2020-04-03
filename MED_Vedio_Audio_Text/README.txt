
### get all early fusion features by concatenation

python EarlyFusion_features.py

### early fusion 1 : MobilenetV2 + soundnet16_kmeans + ASR

python train_kfold_svm.py "P001" "EF_MNetV2_SN16kmeans_ASR_features.npy" 1831 EarlyFusion/EF1_svm.P001.model
python train_kfold_svm.py "P002" "EF_MNetV2_SN16kmeans_ASR_features.npy" 1831 EarlyFusion/EF1_svm.P002.model
python train_kfold_svm.py "P003" "EF_MNetV2_SN16kmeans_ASR_features.npy" 1831 EarlyFusion/EF1_svm.P003.model


### early fusion 2 : MobilenetV2 + soundnet16_gmm + ASR

python train_kfold_svm.py "P001" "EF_MNetV2_SN16gmm_ASR_features.npy" 1831 EarlyFusion/EF2_svm.P001.model
python train_kfold_svm.py "P002" "EF_MNetV2_SN16gmm_ASR_features.npy" 1831 EarlyFusion/EF2_svm.P002.model
python train_kfold_svm.py "P003" "EF_MNetV2_SN16gmm_ASR_features.npy" 1831 EarlyFusion/EF2_svm.P003.model


###Since EF2 gave better ap for validation data, tesing results with EF2

python test_svm.py EarlyFusion/EF2_svm.P001.model "EF_MNetV2_SN16gmm_ASR_features.npy" 1831 EarlyFusion/P001_EF2.lst  "P001"
python test_svm.py EarlyFusion/EF2_svm.P002.model "EF_MNetV2_SN16gmm_ASR_features.npy" 1831 EarlyFusion/P002_EF2.lst  "P002"
python test_svm.py EarlyFusion/EF2_svm.P003.model "EF_MNetV2_SN16gmm_ASR_features.npy" 1831 EarlyFusion/P003_EF2.lst  "P003"


########################################### Double Fusion #######################################################

### train on MobilenetV2
python train_kfold_svm.py "P001" "mobilenetv2_1000_Imgnet_features.npy" 1000 DoubleFusion/MobilenetV2_svm.P001.model
python train_kfold_svm.py "P002" "mobilenetv2_1000_Imgnet_features.npy" 1000 DoubleFusion/MobilenetV2_svm.P002.model
python train_kfold_svm.py "P003" "mobilenetv2_1000_Imgnet_features.npy" 1000 DoubleFusion/MobilenetV2_svm.P003.model

## train on soundnet16_gmm

python train_kfold_svm.py "P001" "50_features_soundnet16.gmm.npy" 50 DoubleFusion/soundnet16_gmm_svm.P001.model
python train_kfold_svm.py "P002" "50_features_soundnet16.gmm.npy" 50 DoubleFusion/soundnet16_gmm_svm.P002.model
python train_kfold_svm.py "P003" "50_features_soundnet16.gmm.npy" 50 DoubleFusion/soundnet16_gmm_svm.P003.model

## train on ASR
python train_kfold_svm.py "P001" "asr_tfidf_features.npy" 781 DoubleFusion/ASR_svm.P001.model
python train_kfold_svm.py "P002" "asr_tfidf_features.npy" 781 DoubleFusion/ASR_svm.P002.model
python train_kfold_svm.py "P003" "asr_tfidf_features.npy" 781 DoubleFusion/ASR_svm.P003.model

##train on MobilenetV2 + soundnet16_gmm
python train_kfold_svm.py "P001" "EF_MNetV2_SN16gmm_features.npy" 1050 DoubleFusion/MNV2_SN16gmm_svm.P001.model
python train_kfold_svm.py "P002" "EF_MNetV2_SN16gmm_features.npy" 1050 DoubleFusion/MNV2_SN16gmm_svm.P002.model
python train_kfold_svm.py "P003" "EF_MNetV2_SN16gmm_features.npy" 1050 DoubleFusion/MNV2_SN16gmm_svm.P003.model

##train on MobilenetV2 + ASR
python train_kfold_svm.py "P001" "EF_MNetV2_ASR_features.npy" 1781 DoubleFusion/MNV2_ASR_svm.P001.model
python train_kfold_svm.py "P002" "EF_MNetV2_ASR_features.npy" 1781 DoubleFusion/MNV2_ASR_svm.P002.model
python train_kfold_svm.py "P003" "EF_MNetV2_ASR_features.npy" 1781 DoubleFusion/MNV2_ASR_svm.P003.model

######## Get scores for event 1 P001
python get_score_svm.py DoubleFusion/MobilenetV2_svm.P001.model "mobilenetv2_1000_Imgnet_features.npy" 1000 DoubleFusion/P001_Score_MV2.lst  "P001"
python get_score_svm.py DoubleFusion/soundnet16_gmm_svm.P001.model "50_features_soundnet16.gmm.npy" 50 DoubleFusion/P001_Score_SN16.lst  "P001"
python get_score_svm.py DoubleFusion/ASR_svm.P001.model "asr_tfidf_features.npy" 781 DoubleFusion/P001_Score_ASR.lst  "P001"
python get_score_svm.py DoubleFusion/MNV2_SN16gmm_svm.P001.model "EF_MNetV2_SN16gmm_features.npy" 1050 DoubleFusion/P001_Score_MV2_SN16.lst  "P001"
python get_score_svm.py DoubleFusion/MNV2_ASR_svm.P001.model "EF_MNetV2_ASR_features.npy" 1781 DoubleFusion/P001_Score_MV2_ASR.lst  "P001"

######## Get scores for event 1 P002
python get_score_svm.py DoubleFusion/MobilenetV2_svm.P002.model "mobilenetv2_1000_Imgnet_features.npy" 1000 DoubleFusion/P002_Score_MV2.lst  "P002"
python get_score_svm.py DoubleFusion/soundnet16_gmm_svm.P002.model "50_features_soundnet16.gmm.npy" 50 DoubleFusion/P002_Score_SN16.lst  "P002"
python get_score_svm.py DoubleFusion/ASR_svm.P002.model "asr_tfidf_features.npy" 781 DoubleFusion/P002_Score_ASR.lst  "P002"
python get_score_svm.py DoubleFusion/MNV2_SN16gmm_svm.P002.model "EF_MNetV2_SN16gmm_features.npy" 1050 DoubleFusion/P002_Score_MV2_SN16.lst  "P002"
python get_score_svm.py DoubleFusion/MNV2_ASR_svm.P002.model "EF_MNetV2_ASR_features.npy" 1781 DoubleFusion/P002_Score_MV2_ASR.lst  "P002"

######## Get scores for event 1 P003
python get_score_svm.py DoubleFusion/MobilenetV2_svm.P003.model "mobilenetv2_1000_Imgnet_features.npy" 1000 DoubleFusion/P003_Score_MV2.lst  "P003"
python get_score_svm.py DoubleFusion/soundnet16_gmm_svm.P003.model "50_features_soundnet16.gmm.npy" 50 DoubleFusion/P003_Score_SN16.lst  "P003"
python get_score_svm.py DoubleFusion/ASR_svm.P003.model "asr_tfidf_features.npy" 781 DoubleFusion/P003_Score_ASR.lst  "P003"
python get_score_svm.py DoubleFusion/MNV2_SN16gmm_svm.P003.model "EF_MNetV2_SN16gmm_features.npy" 1050 DoubleFusion/P003_Score_MV2_SN16.lst  "P003"
python get_score_svm.py DoubleFusion/MNV2_ASR_svm.P003.model "EF_MNetV2_ASR_features.npy" 1781 DoubleFusion/P003_Score_MV2_ASR.lst  "P003"

### comine scores for each event using below script
python DoubleFusion_features.py

### train on DoubleFusion_features
python train_kfold_svm.py "P001" "P001_double_fusion_features.npy" 5 DoubleFusion/Final_doublefusion_svm.P001.model
python train_kfold_svm.py "P002" "P002_double_fusion_features.npy" 5 DoubleFusion/Final_doublefusion_svm.P002.model
python train_kfold_svm.py "P003" "P003_double_fusion_features.npy" 5 DoubleFusion/Final_doublefusion_svm.P003.model

### test on DoubleFusion models
python test_svm.py DoubleFusion/Final_doublefusion_svm.P001.model "P001_double_fusion_features.npy" 1831 DoubleFusion/P001_DF.lst  "P001"
python test_svm.py DoubleFusion/Final_doublefusion_svm.P002.model "P002_double_fusion_features.npy" 1831 DoubleFusion/P002_DF.lst  "P002"
python test_svm.py DoubleFusion/Final_doublefusion_svm.P003.model "P003_double_fusion_features.npy" 1831 DoubleFusion/P003_DF.lst  "P003"