## Multimedia Event Detection

The overview of MED pipeline is depicted in Fig. 1. In the first step, we extract features from the raw training data. The audio features we will use are MFCCs and ASR transcripts
(Additional audio features like the SoundNet features can also be used). The parsing part further does some processing on the extracted raw features to pack them into the final representation, so
that each video is represented by a single feature vector. Implement the bag-of-words representation with k-means clustering. With the representations and the labels, one of the typical approaches to train a classifier is to use Support Vector Machines
(SVMs). The trained SVM models are then to be used in the testing phase. For testing, we extract and pack video representations on the test data with the same parameters/models we use in the
training phase. With the pre-trained classifiers, we then score the videos accordingly and calculate the average precision to evaluate our MED system.

1. run.features.sh generate representations of videos. It extracts features, train k-means(k-words), and represent videos with bag-of-words representation.
2. run.med.sh trains and tests the SVM models and demonstrate the results as APs (Average Precision).