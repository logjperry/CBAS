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
    def __init__(self, window, window_title, video_paths, speed, tsconfig):
        self.window = window
        self.window.title(window_title)
        self.tsconfig = tsconfig
        self.video_paths = video_paths

        self.instance_stack = []

        self.video_index = 0

        # 1 frame of wiggle room
        self.windowl = 5

        # Load video and predictions
        self.cap = cv2.VideoCapture(video_paths[self.video_index])
        self.predictions = pd.read_csv(os.path.splitext(video_paths[self.video_index])[0]+"_predictions.csv")

        self.speed = speed

        self.frame_num = 0

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(window, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        
        # setup the widget for classification
        self.behaviors = [b for b in tsconfig['behaviors']]
        self.behavior = tk.StringVar(value=self.behaviors[0]) 
        self.behaviorselector = ttk.Combobox(window, width = 10, textvariable=self.behavior, values=self.behaviors) 
        self.behaviorselector.pack(anchor=tk.CENTER)

        self.speed_label = tk.Label(window, text='Surfing Speed')
        self.speed_label.pack(anchor=tk.W)

        self.entry_speed = tk.Entry(window,width=5, takefocus=0)
        self.entry_speed.pack(anchor=tk.W)
        self.entry_speed.insert(0, str(self.speed))

        self.board_frame = tk.Frame(window)

        self.scoreboard = Scoreboard(self.board_frame, self.behaviors)
        self.board_frame.pack(anchor=tk.E)

        # Bind arrow keys to functions
        self.window.bind('<Left>', self.prev_frame)
        self.window.bind('<Right>', self.next_frame)

        self.window.bind("<Control-Right>", self.next_video)
        self.window.bind("<Control-Left>", self.previous_video)

        
        self.window.bind("<space>", self.add_instance)
        self.window.bind("<BackSpace>", self.remove_instance)

        
        self.scoreboard.update(self.tsconfig)


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
        self.predictions = pd.read_csv(os.path.splitext(self.video_paths[self.video_index])[0]+"_predictions.csv")
        self.play_video()

    def previous_video(self, event):
        self.cap.release()
        self.video_index+=-1
        self.video_index = self.bound_index(self.video_index)
        self.cap = cv2.VideoCapture(self.video_paths[self.video_index])
        self.predictions = pd.read_csv(os.path.splitext(self.video_paths[self.video_index])[0]+"_predictions.csv")
        self.play_video()

    def add_instance(self, event):

        behavior = self.behavior.get()
        frame = self.get_frame_number()

        instance = {
            'video': self.video_paths[self.video_index],
            'frame': frame,
            'label': behavior,
            'pred': behavior,
            'window':[]
        }

        # check for duplicates

        instances = self.tsconfig['true_instances'][behavior]

        for inst in instances:
            if inst['video']==instance['video'] and inst['frame']==instance['frame']:
                return


        start = frame-self.windowl
        end = frame+self.windowl
        if start<0:
            start = 0
        if end>self.predictions.shape[0]-1:
            end = self.predictions.shape[0]-1
        rows = self.predictions.iloc[int(start+1):int(end+1)].to_numpy()

        pred = self.predictions.iloc[int(frame+1)].to_numpy()
        for i in range(2, len(pred)):
            if pred[i] == 1:
                pred = self.behaviors[i-2]
                break
        instance['pred'] = pred

        preds = []
        for r in rows:
            for i in range(2, len(r)):
                if r[i] == 1:
                    preds.append(self.behaviors[i-2])

        instance['window'] = preds

        self.instance_stack.append(instance)
        self.tsconfig['true_instances'][behavior].append(instance)

        self.scoreboard.update(self.tsconfig)
        self.next_frame(event=None)

    def remove_instance(self, event):
        if len(self.instance_stack) != 0:
            last_inst = self.instance_stack[-1]
            self.instance_stack = self.instance_stack[:-1]

            self.tsconfig['true_instances'][last_inst['label']] = [inst for inst in self.tsconfig['true_instances'][last_inst['label']] if inst['frame'] != last_inst['frame'] or inst['video'] != last_inst['video']]

            self.scoreboard.update(self.tsconfig)
    
class Scoreboard:
    def __init__(self, parent, names):
        self.parent = parent
        self.names = names
        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Name").grid(row=0, column=0)
        tk.Label(self.parent, text="Count").grid(row=0, column=1)
        tk.Label(self.parent, text="Score").grid(row=0, column=2)

        # Store references to the count and score variables
        self.count_vars = {}
        self.score_vars = {}

        # Create rows for each name
        for i, name in enumerate(self.names, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0)

            count_var = tk.StringVar(value="0")  # Initial count
            self.count_vars[name] = count_var
            tk.Label(self.parent, textvariable=count_var).grid(row=i, column=1)

            score_var = tk.StringVar(value="0.0")  # Initial score
            self.score_vars[name] = score_var
            tk.Label(self.parent, textvariable=score_var).grid(row=i, column=2)

    def update(self, config):
        behaviors = config['behaviors']
        instances = config['true_instances']

        for b in behaviors:
            count = len(instances[b])
            self.count_vars[b].set(str(count))

            tp = 0
            fp = 0
            fn = 0

            insts = instances[b]
            for i in insts:
                if i['label'] in i['window']:
                    tp+=1
                elif len(i['window'])!=0:
                    fn+=1

            for b1 in behaviors:
                if b==b1:
                    continue 
                else:
                    insts = instances[b1]
                    for i in insts:
                        if b == i['pred']:
                            fp+=1
            

            if tp+fp==0:
                precision = 1
            else:
                precision = tp/(tp+fp)
            
            if tp+fn==0:
                recall = 1
            else:
                recall = tp/(tp+fn)


            f1_score = int(2*precision*recall/(precision+recall)*1000)/1000
            self.score_vars[b].set(str(f1_score))



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
        
        prediction_file = os.path.splitext(valid_videos[0])[0]+"_predictions.csv"

        df = pd.read_csv(prediction_file)
        # drop the empty column and the background column
        behaviors = (df.columns[2:]).to_numpy()
        instances = {b:[] for b in behaviors}

        tsconfig = {
            'behaviors':behaviors,
            'true_instances':instances,
        }


    speed = 5  # Initial frame skip speed
    RecordingPlayer(tk.Tk(), "Balanced Test Set Creator", valid_videos, speed, tsconfig)
