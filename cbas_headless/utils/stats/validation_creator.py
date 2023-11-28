
import sys
import os
import time
import numpy as np
import json
import yaml
from tabulate import tabulate
import ffmpeg

def get_size(start_path = '.'):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for d in dirnames:
            total_size += 1

    return total_size

def segment_video(video_path, output_path, duration="150"):
    (
        ffmpeg
        .input(video_path)
        .output(output_path, t=duration, c='copy')
        .run()
    )

def concatenate_videos(segment_list, segment_temp_location, output_path):
    with open(segment_temp_location, "w") as file:
        for segment in segment_list:
            file.write(f"file '{segment}'\n")

    ffmpeg.input(segment_temp_location, format='concat', safe=0).output(output_path, c='copy').run()
    os.remove(segment_temp_location)

def make_list(dirname):

    size = get_size(dirname)

    step = np.floor(size/60)
    #step = 8
    i = 2
    video_list = []
    for dir in os.listdir(dirname):
        if not os.path.isdir(dirname+"\\"+dir):
            continue

        if dir.find('_3_')!=-1 or dir.find('_6_')!=-1:
            continue
        
        print(dir)
        if i%step==3:
            video_list.append(dir)
        i+=1
        
    return video_list

if __name__ == "__main__":
    #try:
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
    
    #except:
        #print("Please copy the directory path of the DATA folder and include it as an argument -> python confusion_matrix.py path\DATA")
    