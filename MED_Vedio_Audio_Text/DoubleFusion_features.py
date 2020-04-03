#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import pickle
import sys
import numpy as np
from sklearn import svm
import sklearn.metrics

Score1 = 'DoubleFusion/P001_Score_MV2.lst'
Score2 = 'DoubleFusion/P001_Score_SN16.lst'
Score3 = 'DoubleFusion/P001_Score_ASR.lst'
Score4 = 'DoubleFusion/P001_Score_MV2_SN16.lst'
Score5 = 'DoubleFusion/P001_Score_MV2_ASR.lst'

file1 = open(Score1, "r")
file2 = open(Score2, "r")
file3 = open(Score3, "r")
file4 = open(Score4, "r")
file5 = open(Score5, "r")

score1 = []
for line in file1:
    score1.append(line.strip())

score2 = []
for line in file2:
    score2.append(line.strip())
    
score3 = []
for line in file3:
    score3.append(line.strip())
    
score4 = []
for line in file4:
    score4.append(line.strip())
    
score5 = []
for line in file5:
    score5.append(line.strip())

    
score1 = np.array(score1)
score2 = np.array(score2)
score3 = np.array(score3)
score4 = np.array(score4)
score5 = np.array(score5)

P001_final_score = np.stack((score1, score2, score3, score4, score5))
P001_final_score = P001_final_score.reshape([P001_final_score.shape[1],P001_final_score.shape[0]])

np.save('P001_double_fusion_features.npy', P001_final_score)


Score1 = 'DoubleFusion/P002_Score_MV2.lst'
Score2 = 'DoubleFusion/P002_Score_SN16.lst'
Score3 = 'DoubleFusion/P002_Score_ASR.lst'
Score4 = 'DoubleFusion/P002_Score_MV2_SN16.lst'
Score5 = 'DoubleFusion/P002_Score_MV2_ASR.lst'

file1 = open(Score1, "r")
file2 = open(Score2, "r")
file3 = open(Score3, "r")
file4 = open(Score4, "r")
file5 = open(Score5, "r")

score1 = []
for line in file1:
    score1.append(line.strip())

score2 = []
for line in file2:
    score2.append(line.strip())
    
score3 = []
for line in file3:
    score3.append(line.strip())
    
score4 = []
for line in file4:
    score4.append(line.strip())
    
score5 = []
for line in file5:
    score5.append(line.strip())

    
score1 = np.array(score1)
score2 = np.array(score2)
score3 = np.array(score3)
score4 = np.array(score4)
score5 = np.array(score5)

P002_final_score = np.stack((score1, score2, score3, score4, score5))
P002_final_score = P002_final_score.reshape([P002_final_score.shape[1],P002_final_score.shape[0]])

np.save('P002_double_fusion_features.npy', P002_final_score)

Score1 = 'DoubleFusion/P003_Score_MV2.lst'
Score2 = 'DoubleFusion/P003_Score_SN16.lst'
Score3 = 'DoubleFusion/P003_Score_ASR.lst'
Score4 = 'DoubleFusion/P003_Score_MV2_SN16.lst'
Score5 = 'DoubleFusion/P003_Score_MV2_ASR.lst'

file1 = open(Score1, "r")
file2 = open(Score2, "r")
file3 = open(Score3, "r")
file4 = open(Score4, "r")
file5 = open(Score5, "r")

score1 = []
for line in file1:
    score1.append(line.strip())

score2 = []
for line in file2:
    score2.append(line.strip())
    
score3 = []
for line in file3:
    score3.append(line.strip())
    
score4 = []
for line in file4:
    score4.append(line.strip())
    
score5 = []
for line in file5:
    score5.append(line.strip())

    
score1 = np.array(score1)
score2 = np.array(score2)
score3 = np.array(score3)
score4 = np.array(score4)
score5 = np.array(score5)

P003_final_score = np.stack((score1, score2, score3, score4, score5))
P003_final_score = P003_final_score.reshape([P003_final_score.shape[1],P003_final_score.shape[0]])

np.save('P003_double_fusion_features.npy', P003_final_score)
