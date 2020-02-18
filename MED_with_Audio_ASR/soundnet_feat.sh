#!/bin/bash

for line in $(cat "/home/ubuntu/multimedia/SoundNet-tensorflow/all_file.txt"); do
    echo "${line}" > /home/ubuntu/multimedia/SoundNet-tensorflow/demo.txt
    python /home/ubuntu/multimedia/SoundNet-tensorflow/extract_feat.py -o /home/ubuntu/multimedia/11775-hws-master/hw1_code/sound_net_10 -m 10 -x 11 -s -p extract
done

# Great! We are done!
echo "SUCCESSFUL COMPLETION"
