#!/bin/bash

for line in $(cat "all_file.txt"); do
    echo "${line}" > /home/ubuntu/multimedia/SoundNet-tensorflow/demo.txt
    x=`echo "${line}" | sed 's/data\///' | sed 's/\.mp3//'`
    
    if test -f "/home/ubuntu/multimedia/11775-hws-master/hw1_code/sound_net_16/${x}_16.npy"; then
    	echo "file exists"
    else
        echo "does not exists"
    	python extract_feat.py -o /home/ubuntu/multimedia/11775-hws-master/hw1_code/sound_net_16 -m 16 -x 17 -s -p extract
    	mv /home/ubuntu/multimedia/11775-hws-master/hw1_code/sound_net_16/tf_fea16.npy /home/ubuntu/multimedia/11775-hws-master/hw1_code/sound_net_16/${x}_16.npy
    fi
    
done

# Great! We are done!
echo "SUCCESSFUL COMPLETION"
