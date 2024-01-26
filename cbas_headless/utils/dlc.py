import deeplabcut
import os

def infer(config_path, recording_path, ):

    videos = []

    if os.path.isdir(recording_path):
        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:
                subdir = os.path.join(root, name) 
                video_loc = os.path.join(subdir, name+'.mp4')
                
                if os.path.exists(video_loc):
                    videos.append(video_loc)
    
    deeplabcut.analyze_videos(config_path, videos, shuffle=1, save_as_csv=True, videotype='.mp4', destfolder=storage_path)

