import os
import subprocess
from multiprocessing import Process
import yaml
from sys import exit
import psutil
from datetime import datetime
import time
import signal
import glob
import shutil


def basic_orchestrator(recordingConfig):

    # figure out which cameras to watch
    with open(recordingConfig, 'r') as file:
        rconfig = yaml.safe_load(file)

    modelDict = rconfig['cameras_per_model']
    cameras = rconfig['cameras']

    # modelKey is None for the basic orchestrator
    modelKey = None 
    cam_names_to_watch = modelDict[modelKey]
    cams_to_watch = [cam for cam in cameras if cam['name'] in cam_names_to_watch]

    # get the recording folder
    recordingFolder = os.path.join(os.path.split(recordingConfig)[0], str(modelKey))

    # check to see if it exists already... it shouldn't
    if not os.path.isdir(recordingFolder):
        os.mkdir(recordingFolder)
    
    # make the subfolders for each camera
    for camera in cams_to_watch:
        foldername = os.path.join(recordingFolder, camera['name'])
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

    # loop indefinitely, check for videos that have finished recording. push done videos to destination folder
    while True:
        
        for camera in cams_to_watch:
            # grab the videos that have been recorded from the core/videos/camera location
            video_dir = camera['video_dir']

            real_videos = []

            files = glob.glob(os.path.join(video_dir, '*'))
            for file in files:
                if os.path.isfile(file):
                    real_videos.append(file)
            
            if len(real_videos)==0:
                continue

            real_videos.sort()


            # Don't include the last one
            real_videos = real_videos[:-1]

            finished_videos = []

            # find the finished video names
            subfoldername = os.path.join(recordingFolder, camera['name'])
            for subdir, dirs, files in os.walk(subfoldername):
                for file in files:
                    finished_videos.append(os.path.join(video_dir,file))
            
            # discard videos that have already finished from this list. Do NOT actually delete the real videos yet. That is the job of another process
            real_videos = [vid for vid in real_videos if vid not in finished_videos]


            # Ok, here is where we would usually infer the videos and all that mumbo jumbo, for now, simply push the videos to the 'None' folder
            for video in real_videos:
                video_name = os.path.split(video)[1]
                shutil.copy2(video, os.path.join(subfoldername,video_name))
        
        time.sleep(5)



