## Multimedia Event Detection using Audio

The overview of MED pipeline is depicted in Fig. 1. In the first step, we extract features from the raw training data. Process videos and extract two kinds of visual features
(SURF/CNN) to build video representations.. Implement the bag-of-words representation with k-means clustering. With the representations and the labels, one of the typical approaches to train a classifier is to use Support Vector Machines (SVMs). The trained SVM models are then to be used in the testing phase. For testing, we extract and pack video representations on the test data with the same parameters/models we use in the training phase. With the pre-trained classifiers, we then score the videos accordingly and calculate the average precision to evaluate our MED system.

![MED_Pipeline](https://user-images.githubusercontent.com/46570073/103435242-26275e00-4bda-11eb-8b51-54a52afa4b15.jpg)

## Dataset
It contains 2935 videos, with 3 positive events (P001: assembling shelter; P002: batting in run; P003: making cake) and 1 negative event class (NULL). The data is hosted at S3. Please read README.md for more details. For training, the file all_trn.lst specifies 836 training videos and their labels. For validation, the file all_val.lst contains 400 videos and their ground-truth labels as well. You could use the validation set to tune hyper-parameters, conduct ablation studies and report your interesting findings in the report. For testing, there are additional 1699 videos specified in the all_test_fake.lst, in which their labels are all fake (deliberately set as NULL by us).

## Experimentation and Results
Please see attched pdf(MED_Vedio.pdf) in this same folder to see the experimentations and results in detail. Below is the snapshot of the best results
![Results2](https://user-images.githubusercontent.com/46570073/103435465-caaa9f80-4bdc-11eb-84b8-c6f25c5b59b8.JPG)


## Execution details
Full pipeline is given in the shell script **run.pipeline.sh**.

You can pass pass arguments to this bash script defining which one of the steps (preprocessing: **p**, feature representation: **f**, MAP scores: **m**, kaggle results: **k**, yaml filepath: **y**) you want to perform.

This helps you to avoid rewriting the bash script whenever there are intermediate steps that you don't want to repeat.
Here we also show you how to keep all your parameters in a **yaml file**. It helps to keep track of different parameter configurations that you may try. However, you do not have to keep your parameters in a yaml file. You can change this code as you want.

Here is an example of how to execute the script: 

    bash run.pipeline.sh -p true -f true -m true -k true -y filepath
    
As you already have functions to train kmeans and SVMs, we did not include those skeletons here.
The main TODOs will be to write the function for SURF feature extraction and for CNN feature extraction. **You can reuse your code from HW1 for kmeans and SVM training.**


For Traing and testing:
sh run.med.surf_cnn.sh
