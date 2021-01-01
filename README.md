# Multimedia Event detection
1. Detect events using only the audio features
2. Detect events using only video features
3. Detect events using both audio and video features

### Code and execution details for each of these 3 projects are in respective folders
## For more details on experimentations and results for each part can be found in attached pdf documents in this main folder. 

## Multimedia Event Detection using Audio

The overview of MED pipeline is depicted in Fig. 1. In the first step, we extract features from the raw training data.
For part 1, the audio features we will use are MFCCs and ASR transcripts (Additional audio features like the SoundNet features can also be used).
For part 2, Process videos and extract two kinds of visual features (SURF/CNN) to build video representations.
For part 3, experiment features cobination using early fusion, double fusion and late fusion techniques

Implement the bag-of-words representation with k-means clustering. With the representations and the labels, one of the typical approaches to train a classifier is to use Support Vector Machines (SVMs). The trained SVM models are then to be used in the testing phase. For testing, we extract and pack video representations on the test data with the same parameters/models we use in the training phase. With the pre-trained classifiers, we then score the videos accordingly and calculate the average precision to evaluate our MED system.


![MED_Pipeline](https://user-images.githubusercontent.com/46570073/103435242-26275e00-4bda-11eb-8b51-54a52afa4b15.jpg)

## Dataset
It contains 2935 videos, with 3 positive events (P001: assembling shelter; P002: batting in run; P003: making cake) and 1 negative event class (NULL). The data is hosted at S3. Please read README.md for more details. For training, the file all_trn.lst specifies 836 training videos and their labels. For validation, the file all_val.lst contains 400 videos and their ground-truth labels as well. You could use the validation set to tune hyper-parameters, conduct ablation studies and report your interesting findings in the report. For testing, there are additional 1699 videos specified in the all_test_fake.lst, in which their labels are all fake (deliberately set as NULL by us).

## Early Fusion
As shown in below Figure, for a practical MED system which relies on early fusion for decision, it firstly extracts individual features separately. The extracted features are then combined into a single vector representation for each video. A commonly used feature combination strategy is concatenating vectors from different feature extractors into a long vector. After combination of individual feature vectors for a multimodal representation, the supervised classifiers (such as SVM) are employed for classification.
![EarlyFusion](https://user-images.githubusercontent.com/46570073/103435657-a8fee780-4bdf-11eb-9b60-3f47f13dc5bb.jpg)

## Late Fusion
As illustrated in below Figure, a MED system which uses late fusion for classification also starts with extracting different feature descriptors. In contrast to early fusion, where features are then combined into a multimodal representation, approaches for late fusion firstly learn separate supervised classifiers directly from unimodal features. In the test phase, the prediction scores from different models are then combined to yield a final score. In general, late fusion schemes combine learned unimodal scores into a multimodal representation. Compared to early fusion, late fusion focuses on the individual strength of modalities.
![LateFusion](https://user-images.githubusercontent.com/46570073/103435658-a9977e00-4bdf-11eb-8712-d48fb04dc0e3.jpg)

## Double Fusion
In double fusion, we first perform early fusion to generate different combinations of features from subsets on the single features pool. After that, we train classifiers on each feature or feature combination and carry out late fusion on the output of these classifiers. For example, as shown in below figure, we first extract three kinds of features (visual, audio and text) from three training and three testing videos. After that, pairwise early fusion (visual+audio, visual+text) are carried out in these three features based on their kernel matrices. In the training step, five classifiers are trained based on five features and feature combinations (visual, audio, text, visual+audio, visual+text). For each video, there are thus five output scores indicating how likely it is that this video belongs to the event. In the last step, late fusion is used to fuse five output score vectors into one score vector, on which the final interpretation can be executed.
![DoubleFusion](https://user-images.githubusercontent.com/46570073/103435659-a9977e00-4bdf-11eb-8496-8d42d930550c.jpg)

## Results
Results from feature fusion technique is as below with highlighted result in greeen is the best results:
![results3](https://user-images.githubusercontent.com/46570073/103435677-07c46100-4be0-11eb-8cb1-de8b8b68a1d1.JPG)

## For more details on experimentations and results for each part can be found in attached pdf documents in the main folder. 


