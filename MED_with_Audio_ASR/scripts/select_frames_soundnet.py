#!/bin/python
# Randomly select 

import numpy
import os
import sys

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print ("Usage: {0} file_list select_ratio output_file".format(sys.argv[0]))
        print ("file_list -- the list of video names")
        print ("select_ratio -- the ratio of frames to be randomly selected from each audio file")
        print ("output_file -- path to save the selected frames (feature vectors)")
        exit(1)

    file_list = sys.argv[1]; output_file = sys.argv[3]
    ratio = float(sys.argv[2])

    fread = open(file_list,"r")
    fwrite = open(output_file,"w")

    # random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
    # num_of_frame * ratio rows
    numpy.random.seed(18877)

    for line in fread.readlines():
        feat_path = "./sound_net_16/" + line.replace('\n','') + "_16.npy"
        if os.path.exists(feat_path) == False:
            continue
        array = numpy.load(feat_path)
        numpy.random.shuffle(array)
        select_size = int(array.shape[0] * ratio)
        feat_dim = array.shape[1]

        for n in range(select_size):
            line = str(array[n][0])
            for m in range(1, feat_dim):
                line += ';' + str(array[n][m])
            fwrite.write(line + '\n')
    fwrite.close()

