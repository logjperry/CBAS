
import sys
import os
import time
import numpy as np
import json
import yaml
from tabulate import tabulate
import ffmpeg

def create_set(dirname, exclude=[] set_size=1000):

    i = 0

    video_list = []

    dirlist = [dir for dir in os.listdir(dirname) if os.path.isdir(os.path.join(dirname,dir)) and dir.split('_')[1] not in exclude]
    
    size = len(dirlist)

    sqrt_ss = int(np.ceil(np.sqrt(set_size)))

    step = size/sqrt_ss

    for dir in dirlist:
        if not os.path.isdir(os.path.join(dirname,dir)):
            continue

        cam_name = dir.split('_')[1]
        
        if i%step==0:
            video_list.append(dir)
        i+=1
        
    return video_list

if __name__ == "__main__":
    arg = str(sys.argv[1])   
        
    vl = make_list(arg)
    outputs = []
    for v in vl:
        input_path = os.path.join(os.path.join(arg, v),v+".mp4")
        output_path = os.path.join(os.path.join(arg, 'splice'),v+"_seg.mp4")

        if input_path.find("_4_") >-1:
            continue

        outputs.append(output_path)
        print(input_path)
        if os.path.isfile(input_path):
            print("here")
            segment_video(input_path, output_path)
    concatenate_videos(outputs, os.path.join(os.path.join(arg, 'splice'), 'video_list.txt'), os.path.join(os.path.join(arg, 'splice'), 'compilation.mp4'))
    
    