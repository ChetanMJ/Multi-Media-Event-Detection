# Multimedia Event detection
1. Detect events using only the audio features
2. Detect events using only video features
3. Detect events using both audio and video features

Details for each of these 3 projects are in respective folders

## Multimedia Event Detection using Audio

The overview of MED pipeline is depicted in Fig. 1. In the first step, we extract features from the raw training data.
For part 1, the audio features we will use are MFCCs and ASR transcripts (Additional audio features like the SoundNet features can also be used).
For part 2, Process videos and extract two kinds of visual features (SURF/CNN) to build video representations.
For part 3, experiment features cobination using early fusion, double fusion and late fusion techniques

Implement the bag-of-words representation with k-means clustering. With the representations and the labels, one of the typical approaches to train a classifier is to use Support Vector Machines (SVMs). The trained SVM models are then to be used in the testing phase. For testing, we extract and pack video representations on the test data with the same parameters/models we use in the training phase. With the pre-trained classifiers, we then score the videos accordingly and calculate the average precision to evaluate our MED system.


![MED_Pipeline](https://user-images.githubusercontent.com/46570073/103435242-26275e00-4bda-11eb-8b51-54a52afa4b15.jpg)
