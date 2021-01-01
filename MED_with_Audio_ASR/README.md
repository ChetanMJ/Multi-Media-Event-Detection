## Multimedia Event Detection

The overview of MED pipeline is depicted in Fig. 1. In the first step, we extract features from the raw training data. The audio features we will use are MFCCs and ASR transcripts
(Additional audio features like the SoundNet features can also be used). The parsing part further does some processing on the extracted raw features to pack them into the final representation, so
that each video is represented by a single feature vector. Implement the bag-of-words representation with k-means clustering. With the representations and the labels, one of the typical approaches to train a classifier is to use Support Vector Machines
(SVMs). The trained SVM models are then to be used in the testing phase. For testing, we extract and pack video representations on the test data with the same parameters/models we use in the
training phase. With the pre-trained classifiers, we then score the videos accordingly and calculate the average precision to evaluate our MED system.

1. run.features.sh generate representations of videos. It extracts features, train k-means(k-words), and represent videos with bag-of-words representation.
2. run.med.sh trains and tests the SVM models and demonstrate the results as APs (Average Precision).

![MED_Pipeline](https://user-images.githubusercontent.com/46570073/103435242-26275e00-4bda-11eb-8b51-54a52afa4b15.jpg)

## Dataset
It contains 2935 videos, with 3 positive events (P001: assembling shelter; P002: batting in run; P003: making cake) and 1 negative event class (NULL). The data is hosted at S3. Please read README.md for more details. For training, the file all_trn.lst specifies 836 training videos and their labels. For validation, the file all_val.lst contains 400 videos and their ground-truth labels as well. You could use the validation set to tune hyper-parameters, conduct ablation studies and report your interesting findings in the report. For testing, there are additional 1699 videos specified in the all_test_fake.lst, in which their labels are all fake (deliberately set as NULL by us).


