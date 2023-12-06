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
from tkinter import ttk

class RecordingPlayer:
    def __init__(self, window, window_title, video_paths, speed, tsconfig, config_path):
        self.window = window
        self.window.title(window_title)
        self.tsconfig = tsconfig
        self.video_paths = video_paths
        self.config_path = config_path
        self.instance_stack = []

        self.instance_cache = []
        self.video_index = 0
        
        # setup the widget for classification
        self.behaviors = [b for b in tsconfig['behaviors']]
        self.update_instance_cache()

        self.color_list = ['coral','goldenrod1','royalblue','red1','deeppink1','lightslateblue','gray23','limegreen']


        # 1 frame of wiggle room
        self.windowl = 5

        # Load video and predictions
        self.cap = cv2.VideoCapture(video_paths[self.video_index])
        
        self.speed = speed

        self.frame_num = 0

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(column=0, row=0)

        self.start = -1 
        self.end = -1

        # self.startButton = tk.Button(window, text='Start', command=self.start_frame).pack(anchor=tk.CENTER)
        # self.endButton = tk.Button(window, text='End', command=self.end_frame).pack(anchor=tk.CENTER)

        

        
        self.board_frame = tk.Frame(window)

        self.scoreboard = Scoreboard(self.board_frame, self.behaviors, self.color_list)
        self.board_frame.grid(column=1, row=0)
    
        self.ss = tk.StringVar(value='Surf Speed = '+str(self.speed))
        self.speed_label = tk.Label(window, textvariable=self.ss, pady=5)
        self.speed_label.grid(column=0, row=1)


        # Bind arrow keys to functions
        self.window.bind('<Left>', self.prev_frame)
        self.window.bind('<Right>', self.next_frame)
        
        self.window.bind('<Up>', self.incr_ss)
        self.window.bind('<Down>', self.decr_ss)

        self.window.bind("<Control-Right>", self.next_video)
        self.window.bind("<Control-Left>", self.previous_video)

        self.window.bind("<BackSpace>", self.remove_instance)

        

        for i in range(1,len(self.behaviors)+1):
            self.window.bind(str(i), lambda i: self.start_frame(i))

        self.primed = False
        
        self.scoreboard.update(self.tsconfig)


        # Update & delay variables
        self.delay = 15   # ms
        self.update()

        self.play_video()

        self.window.mainloop()
    
    def update_instance_cache(self):
        self.instance_cache = [inst for b in self.behaviors for inst in self.tsconfig['instances'][b] if inst['video']==self.video_paths[self.video_index]]

    def play_video(self):
        # Read the next frame from the video. If reached the end of the video, release the video capture object
        ret, frame = self.cap.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            if self.primed:
                self.canvas.create_oval(10, 10, 30, 30, fill=self.color_list[self.behavior_index-1], outline="#DDD", width=1)
            
            # check to see if frame is in an instance range
            behaviors = self.behaviors
            instances = self.tsconfig['instances']
            frame = self.get_frame_number()

            for b in behaviors:
                video_found = False
                for inst in instances[b]:
                    if inst['video']!=self.video_paths[self.video_index]:
                        continue 
                    elif inst['start']<=frame and inst['end']>=frame:
                        ind = self.behaviors.index(inst['label'])
                        self.canvas.create_oval(10, 10, 30, 30, fill=self.color_list[ind], outline="#DDD", width=1)
                    else:
                        video_found = True
                        continue
                if video_found:
                    self.canvas.create_rectangle(10, 230, 30, 250, fill='black', outline="#DDD", width=1)

    def incr_ss(self, event):
        self.speed*=2
        self.ss.set(value='Surf Speed = '+str(self.speed))
    def decr_ss(self, event):
        if self.speed<=1:
            self.speed = 1
        else:
            self.speed = int(round(self.speed/2))
        
        self.ss.set(value='Surf Speed = '+str(self.speed))
    def update(self):
        # Get the latest value for speed
        try:
            self.speed = int(self.speed)
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
    
    def bound_index(self, index):
        if index<len(self.video_paths):
            if index>0:
                return index 
            else:
                return 0
        else:
            return len(self.video_paths)-1

    def next_video(self, event):
        self.cap.release()
        self.video_index+=1
        self.video_index = self.bound_index(self.video_index)
        self.cap = cv2.VideoCapture(self.video_paths[self.video_index])
        self.update_instance_cache()
        self.play_video()

    def previous_video(self, event):
        self.cap.release()
        self.video_index+=-1
        self.video_index = self.bound_index(self.video_index)
        self.cap = cv2.VideoCapture(self.video_paths[self.video_index])
        self.update_instance_cache()
        self.play_video()
    
    def isoverlap(self, s1, e1, s2, e2):
        return max(s1, s2) <= min(e1, e2)
            

    def add_instance(self, behavior):

        instances = self.tsconfig['instances']

        video = self.video_paths[self.video_index]
        start = self.start 
        end = self.end


        if end < start:
            temp = start 
            start = end 
            end = temp 
        
        if end-start < 10:
            print('Range should be at least 10 frames!')
            return
        

        if start==-1 or end==-1:
            print('Finish defining the boundaries!')
            return 

        # check for duplicates
        for inst in instances[behavior]:

            if inst['video'] == video and self.isoverlap(start, end, inst['start'],inst['end']):
                print('Overlapping behavior region!')
                return

        instance = {
            'label':behavior,
            'video':video,
            'start':start,
            'end':end,
            'length':(end-start)
        }

        self.tsconfig['instances'][behavior].append(instance)

        self.instance_stack.append(instance)

        self.start = -1 
        self.end = -1

        self.scoreboard.update(self.tsconfig)

    def start_frame(self, index):
        if self.primed:
            self.end_frame(index)
            self.primed = False
            return
        self.behavior_index = int(index.char)
        self.start = self.get_frame_number()
        self.primed = True
    def end_frame(self, index):
        index = int(index.char)
        if self.behavior_index!=index:
            print('Behavior index does not match previously selected index.')
            return
        if index-1 >= 0 and index-1 < len(self.behaviors):
            self.behavior = self.behaviors[index-1]
        else:
            print('Behavior index out of bounds.')
            return
        self.end = self.get_frame_number()
        self.add_instance(behavior=self.behaviors[index-1])

    def remove_instance(self, event):
        if len(self.instance_stack) != 0:
            last_inst = self.instance_stack[-1]
            self.instance_stack = self.instance_stack[:-1]

            self.tsconfig['instances'][last_inst['label']] = [inst for inst in self.tsconfig['instances'][last_inst['label']] if inst['start'] != last_inst['start'] or inst['video'] != last_inst['video'] or inst['end'] != last_inst['end']]

            self.scoreboard.update(self.tsconfig)
    
class Scoreboard:
    def __init__(self, parent, names, colors):
        self.parent = parent
        self.names = names
        self.colors = colors
        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0)
        tk.Label(self.parent, text="Index", font=('Arial',9,'bold','underline')).grid(row=0, column=1)
        tk.Label(self.parent, text="Frames", font=('Arial',9,'bold','underline')).grid(row=0, column=2)

        # Store references to the count and score variables
        self.count_vars = {}

        # Create rows for each name
        for i, name in enumerate(self.names, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0)

            # index
            tk.Label(self.parent, text=str(i), bg=self.colors[i-1]).grid(row=i, column=1, pady=3)

            count_var = tk.StringVar(value="0")  # Initial count
            self.count_vars[name] = count_var
            tk.Label(self.parent, textvariable=count_var).grid(row=i, column=2)



    def update(self, config):
        behaviors = config['behaviors']
        instances = config['instances']

        for b in behaviors:
            count = 0
            for inst in instances[b]:
                count += inst['length']
            self.count_vars[b].set(str(count))


def save_config(tsconfig, config_path):
    with open(config_path, 'w+') as file:
        yaml.dump(tsconfig, file, allow_unicode=True)


# This function is just expecting a directory with subdirectories containing mp4 videos and labels in the DEG form
def create_training_set(recording_path, behaviors=[], project_config='undefined'):

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

        print(test_sets)
    
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

    recording_name = os.path.split(recording_path)[1]
    training_set_config = os.path.join(test_sets,recording_name+"_training.yaml") 

    if os.path.exists(training_set_config):
        # load the preexisting test_set config
        with open(training_set_config, 'r') as file:
            tsconfig = yaml.safe_load(file)

        # make sure that the recording folder has what we need in it
        valid_videos = []

        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:
                subdir = os.path.join(root, name) 

                video_loc = os.path.join(subdir, name+'.mp4')


                if not os.path.exists(video_loc):
                    print(f'{subdir} does match the following structure. \n/Recording_Directory \n\t/subdirectory \n\t\t/subdirectory.mp4\n.\n.\n.')
                    continue
                valid_videos.append(video_loc)

        # grab a valid prediction file and get the behaviors from it
        if len(valid_videos)==0:
            raise Exception('No valid videos found. Exiting.')
        
        behaviors = tsconfig['behaviors']
        instances = tsconfig['instances']
        for b in behaviors:
            for inst in instances[b]:
                if inst['video'] in valid_videos:
                    continue 
                else:
                    vid = inst['video']
                    raise Exception(f'{vid} not found in the recording folder. This video contains behavior instances and needs to be included in the recording folder.')
    else:
        
        # make sure that the recording folder has what we need in it
        valid_videos = []

        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:
                subdir = os.path.join(root, name) 

                video_loc = os.path.join(subdir, name+'.mp4')


                if not os.path.exists(video_loc):
                    print(f'{subdir} does match the following structure. \n/Recording_Directory \n\t/subdirectory \n\t\t/subdirectory.mp4\n.\n.\n.')
                    continue
                valid_videos.append(video_loc)

        # grab a valid prediction file and get the behaviors from it
        if len(valid_videos)==0:
            raise Exception('No valid videos found. Exiting.')
        
        # open a small pop-up to prompt the user for the behaviors
        if len(behaviors)==0:
            raise Exception('You must enter at least one behavior.')
        instances = {b:[] for b in behaviors}

        tsconfig = {
            'behaviors':behaviors,
            'instances':instances,
        }


    speed = 5  # Initial frame skip speed
    root = tk.Tk()
    rp = RecordingPlayer(root, "Training Set Creator", valid_videos, speed, tsconfig, training_set_config)

    save_config(rp.tsconfig, rp.config_path)

