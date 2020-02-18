#!/bin/bash

for line in $(cat "list/all.video"); do
    ffmpeg -i /home/ubuntu/multimedia/11775-hws-master/videos/${line}.mp4 -vn -acodec libmp3lame -ac 2 -ab 160k -ar 48000 ${line}.mp3
    sox ${line}.mp3 /home/ubuntu/multimedia/SoundNet-tensorflow/data/${line}.mp3 trim 0
    echo "data/${line}.mp3" >> /home/ubuntu/multimedia/SoundNet-tensorflow/all_files.txt
    rm -f /home/ubuntu/multimedia/11775-hws-master/videos/${line}.mp4
    rm -f ${line}.mp3
done

# Great! We are done!
echo "SUCCESSFUL COMPLETION"
