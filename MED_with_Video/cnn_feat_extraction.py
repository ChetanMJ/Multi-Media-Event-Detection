#!/usr/bin/env python3

import os
import sys
import threading
import cv2
import numpy as np
import yaml
import pickle
import pdb
from PIL import Image
import torch
from PIL import Image
from torchvision import transforms

# generate random integer values
from random import seed
from random import randint
# seed random number generator
seed(1)


def get_cnn_features_from_video(downsampled_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    
    downsampled_video_frames = get_keyframes(downsampled_video_filename, 10)
    z=list(downsampled_video_frames)
    max_len = len(z)
    #print(max_len)
    z=np.array(z)
    #print(z.shape)
    
    # generate some integers
    
    random_index=[]
    
    for _ in range(6):
    	value = randint(0, max_len-1)
    	random_index.append(value)

    
    #print(random_index)
    downsampled_video_frames = z[np.array(random_index)]
    
    key_frame_cnn_feature = []
    
    for frame_index in range(len(downsampled_video_frames)):
    	im = Image.fromarray(downsampled_video_frames[frame_index])
    	im.save("temp.jpeg")
    	input_image = Image.open("temp.jpeg")
    	preprocess = transforms.Compose([
    	transforms.Resize(256),
    	transforms.CenterCrop(224),
    	transforms.ToTensor(),
    	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    	])
    	input_tensor = preprocess(input_image)
    	input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
 
    	# move the input and model to GPU for speed if available
    	if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')
            
    	with torch.no_grad():
            output = model(input_batch)        

    	# The output has unnormalized scores. To get probabilities, you can run a softmax on it.
    	a = torch.nn.functional.softmax(output[0], dim=0)
    	a = a.cpu().numpy()
    	key_frame_cnn_feature.append(a)
	
    key_frame_cnn_feature = np.array(key_frame_cnn_feature)
    key_frame_cnn_feature = np.average(key_frame_cnn_feature,axis=0)
    return key_frame_cnn_feature
	


def get_keyframes(downsampled_video_filename, keyframe_interval):
    "Generator function which returns the next keyframe."

    # Create video capture object
    video_cap = cv2.VideoCapture(downsampled_video_filename)
    frame = 0
    while True:
        frame += 1
        ret, img = video_cap.read()
        if ret is False:
            break
        if frame % keyframe_interval == 0:
            yield img
    video_cap.release()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage {0} video_list config_file".format(sys.argv[0]))
        print("video_list -- file containing video names")
        print("config_file -- yaml filepath containing all parameters")

    all_video_names = sys.argv[1]
    config_file = sys.argv[2]
    config_file="/home/ubuntu/multimedia/11775-hws-master/hw2_code/config.yaml"
    my_params = yaml.load(open(config_file), Loader=yaml.FullLoader)

    # Get parameters from config file
    downsampled_videos = my_params.get('downsampled_videos')
    cnn_feature_path="/home/ubuntu/multimedia/11775-hws-master/hw2_code/cnn"

    # Create pretrained cnn model
    model = torch.hub.load('pytorch/vision:v0.5.0', 'mobilenet_v2', pretrained=True)
    model.eval()


    mobilnetv2_feature_index = np.array([672,472,915,656,436,518,608,580,415,449,896,923,720,868,712,899,898,968,521,572,647,618,469,659,438,651,544,827,733,981,693,615,429,977])
    mobilnetv2_feature_index = np.sort(mobilnetv2_feature_index)

    # Check if folder for SURF features exists
    if not os.path.exists(cnn_feature_path):
        os.mkdir(cnn_feature_path)

    # Loop over all videos (training, val, testing)
    # TODO: get mobilenetv2 cnn features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    all_cnn_features = []
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        print(video_name)
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        #outfile = '/home/ubuntu/multimedia/11775-hws-master/hw2_code/cnn/'+video_name+'_cnn.npy'
        #if os.path.isfile(outfile):
            #print(video_name+" File exists")
            #continue


        if not os.path.isfile(downsampled_video_filename):
            y = np.zeros(len(mobilnetv2_feature_index))
            all_cnn_features.append(y)
            np.save(outfile,y)
            continue

        # Get SURF features for one video
        cnn_feature = get_cnn_features_from_video(downsampled_video_filename,keyframe_interval=10)
        #cnn_feature = cnn_feature[mobilnetv2_feature_index]
        all_cnn_features.append(cnn_feature)
        #np.save(outfile,cnn_feature)
        
    all_cnn_features = np.array(all_cnn_features)
    np.save("mobilenetv2_1000_Imgnet_features",all_cnn_features)
    
