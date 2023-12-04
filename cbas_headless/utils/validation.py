import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
import os
import time
import ctypes
import yaml
import os
import math
import sys
from sys import exit
import subprocess
import shutil
import pandas as pd

class RecordingPlayer:
    def __init__(self, window, window_title, video_paths, speed):
        self.window = window
        self.window.title(window_title)

        # Load video
        self.cap = cv2.VideoCapture(video_paths[0])
        self.speed = speed

        self.frame_num = 0

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.speed_label = tk.Label(window, text='Surfing Speed')
        self.speed_label.pack(anchor=tk.W)

        self.entry_speed = tk.Entry(window,width=5)
        self.entry_speed.pack(anchor=tk.W)
        self.entry_speed.insert(0, str(self.speed))

        # Bind arrow keys to functions
        self.window.bind('<Left>', self.prev_frame)
        self.window.bind('<Right>', self.next_frame)

        # Update & delay variables
        self.delay = 15   # ms
        self.update()

        self.play_video()

        self.window.mainloop()

    def play_video(self):
        # Read the next frame from the video. If reached the end of the video, release the video capture object
        ret, frame = self.cap.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def update(self):
        # Get the latest value for speed
        try:
            self.speed = int(self.entry_speed.get())
        except ValueError:
            self.speed = 1
            self.entry_speed.delete(0, tk.END)
            self.entry_speed.insert(0, '1')

        self.window.after(self.delay, self.update)

    def next_frame(self, event):
        # Move to next frame
        frame_no = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no + self.speed)
        self.play_video()

    def prev_frame(self, event):
        # Move to previous frame
        frame_no = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - self.speed))
        self.play_video()

    def get_frame_number(self):
        frame_no = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return frame_no



# This function is just expecting a directory with subdirectories containing mp4 videos and labels in the DEG form
def create_balanced_set(recording_path, project_config='undefined'):

    # open the project config and get the test_set yaml path
    if project_config=='undefined':
        # assume that the user is located in an active project
        user_dir = os.getcwd()

        # make sure user is located within the main directory of a project
        project_config = os.path.join(user_dir, 'project_config.yaml')

        if os.path.exists(project_config):
            print('Project found.')
        else:
            raise Exception('Project not found.')
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the project config file. Check for yaml syntax errors.')

        # grabbing the locations of the test_sets
        test_sets = pconfig['test_sets_path']
    
    else:

        if os.path.exists(project_config):
            print('Project found.')
        else:
            raise Exception('Project not found.')
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the project config file. Check for yaml syntax errors.')

        # grabbing the locations of the test_sets
        test_sets = pconfig['test_sets_path']

    recording_name = os.path.split(recording_path)[0]
    test_set_config = os.path.join(test_sets,recording_name+".yaml") 

    if os.path.exists(test_set_config):
        # load the preexisting test_set config
        with open(test_set_config, 'r') as file:
            tsconfig = yaml.safe_load(file)
    else:
        
        # make sure that the recording folder has what we need in it
        valid_videos = []

        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:
                subdir = os.path.join(root, name) 

                video_loc = os.path.join(subdir, name+'.mp4')
                prediction_loc = os.path.join(subdir, name+'_predictions.csv')


                if not os.path.exists(video_loc) or not os.path.exists(prediction_loc):
                    print(f'{subdir} does match the following structure. \n/Recording_Directory \n\t/subdirectory \n\t\t/subdirectory.mp4 \n\t\t/subdirectory_predictions.csv \n.\n.\n.')
                    continue
                valid_videos.append(video_loc)

        # grab a valid prediction file and get the behaviors from it
        if len(valid_videos)==0:
            raise Exception('No valid videos found. Exiting.')
        


        tsconfig = {
            'behaviors':[],
            'true_instances':[]
        }

    speed = 5  # Initial frame skip speed
    RecordingPlayer(tk.Tk(), "Balanced Test Set Creator", valid_videos, speed, tsconfig)
