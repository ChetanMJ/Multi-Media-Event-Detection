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


def get_surf_features_from_video(downsampled_video_filename, surf_feat_video_filename, keyframe_interval):
    "Receives filename of downsampled video and of output path for features. Extracts features in the given keyframe_interval. Saves features in pickled file."
    
    downsampled_video_frames = get_keyframes(downsampled_video_filename, 10)
    downsampled_video_frames = list(downsampled_video_frames)[:6]
    
    for frame_index in range(len(downsampled_video_frames)):
    	im = Image.fromarray(downsampled_video_frames[frame_index])
    	im.save("temp.jpeg")
    	image = cv2.imread("temp.jpeg", cv2.IMREAD_GRAYSCALE) #Load the image to grayscale color
    	##surf = cv2.xfeatures2d.SURF_create(100)
    	kp, disc = surf.detectAndCompute(image, None)
    	np.save(surf_feat_video_filename+"_"+str(frame_index), disc)
    	os.remove("temp.jpeg")


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
    keyframe_interval = my_params.get('keyframe_interval')
    hessian_threshold = my_params.get('hessian_threshold')
    surf_features_folderpath = my_params.get('surf_features')
    downsampled_videos = my_params.get('downsampled_videos')

    # TODO: Create SURF object
    surf = cv2.xfeatures2d.SURF_create(hessian_threshold)

    # Check if folder for SURF features exists
    if not os.path.exists(surf_features_folderpath):
        os.mkdir(surf_features_folderpath)

    # Loop over all videos (training, val, testing)
    # TODO: get SURF features for all videos but only from keyframes

    fread = open(all_video_names, "r")
    for line in fread.readlines():
        video_name = line.replace('\n', '')
        downsampled_video_filename = os.path.join(downsampled_videos, video_name + '.ds.mp4')
        surf_feat_video_filename = os.path.join(surf_features_folderpath, video_name + '.surf')

        if not os.path.isfile(downsampled_video_filename):
            continue

        # Get SURF features for one video
        get_surf_features_from_video(downsampled_video_filename,
                                     surf_feat_video_filename, keyframe_interval)
