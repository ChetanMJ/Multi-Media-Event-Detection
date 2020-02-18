#!/bin/python
import numpy
import numpy as np
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys


import re
import string
import codecs
import os.path
from sklearn.feature_extraction.text import TfidfVectorizer


## funtion to remove punctuations and get list of words
def make_word_list(sentence):
    
    for c in string.punctuation:
        corpus_text = sentence.replace(c, "") 
    text = re.sub(r'\S*\d\S*', '', corpus_text)
    text = re.sub(r'[^\w\s]', '', text) 
    text = text.split()   
    li = []
    for token in text:
        li.append(token)

    return " ".join(li)



if __name__ == '__main__':
    if len(sys.argv) != 3:
        print ("Usage: {0} vocab_file, file_list".format(sys.argv[0]))
        print ("vocab_file -- path to the vocabulary file")
        print ("file_list -- the list of videos")
        
        
        ##combine all asr of vedios into single list
        ##if no asr then insert blank into this list
        
        file_list = './list/all.video'
        f = open(file_list, "r")
        all_asr = []
        for file in f:
            file = file.strip()
            file_name = '../asrs/'+file+'.txt'
            if os.path.isfile(file_name):
                file_asr=open(file_name, "r")
                sentence = ''
                for fi in file_asr:
                    sentence = sentence +' '+ fi.strip()
                all_asr.append(sentence)
            else:
                all_asr.append('')


	## convert raw asr data into list of words
        sentences=[]
        for asr in all_asr:
            sentences.append(make_word_list(asr))
	

	## convert list of words for each video to vector using tfidf vectorizer
        vectorizer = TfidfVectorizer(stop_words="english", max_df=0.8, max_features=1000, min_df=10)
        X = vectorizer.fit_transform(sentences)

    	##save the vector
        out_file = './asrfeat/asr_tfidf_features.npy'
        np.save(out_file, X.toarray())
    	
    

    print ("ASR features generated successfully!")
