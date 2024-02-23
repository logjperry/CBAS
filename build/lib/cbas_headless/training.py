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
import numpy as np
import random
import pickle


from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix

import ttkbootstrap as ttk
theme = 'superhero'

class RecordingPlayer:
    def __init__(self, window, window_title, video_paths, speed, tsconfig, config_path, color_list=['coral','goldenrod1','royalblue','red1','deeppink1','lightslateblue','gray23','limegreen','cornsilk']):
        self.window = window
        self.window.title(window_title)
        self.tsconfig = tsconfig
        self.video_paths = video_paths
        self.config_path = config_path
        self.instance_stack = []


        self.instance_cache = []
        self.video_index = 0

        self.upper = tk.Frame(self.window)

        
        # setup the widget for classification
        self.behaviors = [b for b in tsconfig['behaviors']]
        self.update_instance_cache()

        self.color_list = color_list


        # Load video and predictions
        self.cap = cv2.VideoCapture(video_paths[self.video_index])
        
        self.speed = speed

        self.frame_num = 0

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.upper, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(column=0, row=0)

        self.start = -1 
        self.end = -1

        # self.startButton = tk.Button(window, text='Start', command=self.start_frame).pack(anchor=tk.CENTER)
        # self.endButton = tk.Button(window, text='End', command=self.end_frame).pack(anchor=tk.CENTER)

        

        
        self.board_frame = tk.Frame(self.upper)
        self.scoreboard = Scoreboard(self.board_frame, self.behaviors, self.color_list)
        self.board_frame.grid(column=1, row=0, padx=10, sticky="n")
    
        self.ss = tk.StringVar(value='Surf Speed = '+str(self.speed))
        self.speed_label = tk.Label(self.upper, textvariable=self.ss, pady=5, font=('TkDefaultFixed', 10))
        self.speed_label.grid(column=0, row=1)

        self.upper.pack(anchor=tk.CENTER, pady=10, padx=10)

        self.shortcuts = tk.Frame(self.window)
        tk.Label(self.shortcuts, text='Use <Left> and <Right> arrows to surf through video\nUse <Ctrl> + <Left>/<Right> to switch videos\nUse <Up> and <Down> arrows to change surf speed\nUse index numbers to start/stop labeling frames as a behavior\nUse <Back> to remove a label\nSimply exit the GUI to save', justify=tk.LEFT, pady=5, font=('TkDefaultFixed', 10)).pack()
        self.shortcuts.pack(anchor='e',side='bottom', pady=10, padx=10)

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

    def resize_image(self, image, width, height):
        # Resize image maintaining aspect ratio
        img_aspect = image.width / image.height
        win_aspect = width / height

        if img_aspect > win_aspect:
            new_width = width
            new_height = int(width / img_aspect)
        else:
            new_width = int(height * img_aspect)
            new_height = height

        return image.resize((new_width, new_height), Image.ANTIALIAS)

    def play_video(self):
        # Read the next frame from the video. If reached the end of the video, release the video capture object
        ret, frame = self.cap.read()
        if ret:
            self.canvas.delete("all")
            self.image = self.resize_image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), 500, 500)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.config(width=self.image.width, height=self.image.height)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            if self.primed:
                self.canvas.create_oval(10, 10, 50, 50, fill=self.color_list[self.behavior_index-1], outline="#000", width=2)

            # check to see if frame is in an instance range
            behaviors = self.behaviors
            instances = self.tsconfig['instances']
            frame = self.get_frame_number()
            video_found = False


            if len(self.instance_cache)==0:
                video_found = False
            else:
                video_found = True

            for inst in self.instance_cache:
                if inst['video']!=self.video_paths[self.video_index]:
                    continue 
                elif inst['start']<=frame and inst['end']>=frame:
                    ind = self.behaviors.index(inst['label'])
                    self.canvas.create_oval(10, 10, 50, 50, fill=self.color_list[ind], outline="#000", width=2)
                else:
                    video_found = True
                    continue

            if video_found:
                self.canvas.create_rectangle(10, self.image.height-50, 50, self.image.height-10, fill='black', outline="#DDD", width=1)

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

        self.update_instance_cache()

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
            self.update_instance_cache()

class Scoreboard:
    def __init__(self, parent, names, colors):
        self.parent = parent
        self.names = names
        self.colors = colors
        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Behavior", font=('TkDefaultFixed', 15,'underline')).grid(row=0, column=0)
        tk.Label(self.parent, text="Index", font=('TkDefaultFixed', 15,'underline')).grid(row=0, column=1)
        tk.Label(self.parent, text="Frames", font=('TkDefaultFixed', 15,'underline')).grid(row=0, column=2)
        tk.Label(self.parent, text="Instances", font=('TkDefaultFixed', 15,'underline')).grid(row=0, column=3)

        # Store references to the count and score variables
        self.count_vars = {}
        self.inst_vars = {}

        # Create rows for each name
        for i, name in enumerate(self.names, start=1):
            tk.Label(self.parent, text=name, font=('TkDefaultFixed', 12)).grid(row=i, column=0)

            # index
            tk.Label(self.parent, text=str(i), bg=self.colors[i-1], font=('TkDefaultFixed', 12), autostyle=False).grid(row=i, column=1, pady=3)

            count_var = tk.StringVar(value="0")  # Initial count
            self.count_vars[name] = count_var
            tk.Label(self.parent, textvariable=count_var, font=('TkDefaultFixed', 12)).grid(row=i, column=2)

            inst_var = tk.StringVar(value="0")  # Initial instances
            self.inst_vars[name] = inst_var
            tk.Label(self.parent, textvariable=inst_var, font=('TkDefaultFixed', 12)).grid(row=i, column=3)



    def update(self, config):
        behaviors = config['behaviors']
        instances = config['instances']

        for b in behaviors:
            count = 0
            for inst in instances[b]:
                count += inst['length']
            self.count_vars[b].set(str(count))
            self.inst_vars[b].set(str(len(instances[b])))

# Saves the training set config
def save_config(tsconfig, config_path):
    """
    Save the training set configuration to a yaml file.
    
    Parameters:
    -----------
    tsconfig: dict
        The training set configuration dictionary
    config_path: str
        The path to save the training set configuration

    """
    with open(config_path, 'w+') as file:
        yaml.dump(tsconfig, file, allow_unicode=True)

# This function is just expecting a directory with subdirectories containing mp4 videos and labels in the DEG form
def create_training_set(recording_name, behaviors=[], project_config='undefined'):

    """
    Create a training set from a recording directory. The recording directory should contain subdirectories with mp4 videos in the CBAS form. 

    Parameters:
    -----------
    recording_name: str
        The name of the recording directory
    behaviors: list
        A list of behaviors to include in the training set
    project_config: str
        The path to the project config file. If 'undefined', the function will assume that the user is located within an active project.
    """

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
        training_set = pconfig['trainingsets_path']
        recordings_path = pconfig['recordings_path']

    
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
        training_set = pconfig['trainingsets_path']
        recordings_path = pconfig['recordings_path']

    recording_path = os.path.join(recordings_path, recording_name,'videos')
    training_set_config = os.path.join(training_set,recording_name+"_training.yaml") 
    recording_config = os.path.join(recordings_path, recording_name, 'details.yaml')

    rconfig = None
    if os.path.exists(recording_config):
        with open(recording_config, 'r') as file:
            rconfig = yaml.safe_load(file)

    if os.path.exists(training_set_config):
        # load the preexisting test_set config
        with open(training_set_config, 'r') as file:
            tsconfig = yaml.safe_load(file)

        # make sure that the recording folder has what we need in it
        valid_videos = []

        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:
                subdir = os.path.join(root, name) 

                vids = rconfig['cameras_files'][name]
                for v in vids:
                    video_loc = os.path.join(subdir, list(v.keys())[0])


                    if not os.path.exists(video_loc):
                        print(f'{subdir} does match the following structure. \n/Recording_Directory \n\t/cameraname \n\t\t/recording.mp4\n.\n.\n.')
                        continue
                    valid_videos.append(video_loc)

        if len(valid_videos)==0:
            raise Exception('No valid videos found. Exiting.')
        
        behaviors_stored = tsconfig['behaviors']
        instances_stored  = tsconfig['instances']

        for b in behaviors:
            if b not in behaviors_stored:
                behaviors_stored.append(b)
                instances_stored[b]=[]

        for b in behaviors_stored:
            for inst in instances_stored[b]:
                if inst['video'] in valid_videos:
                    continue 
                else:
                    vid = inst['video']
                    raise Exception(f'{vid} not found in the recording folder. This video contains behavior instances and needs to be included in the recording folder.')
        
        behaviors = behaviors_stored
        instances = instances_stored
    else:
        
        # make sure that the recording folder has what we need in it
        valid_videos = []

        for root, dirs, files in os.walk(recording_path, topdown=False):
            for name in dirs:

                subdir = os.path.join(root, name) 

                vids = rconfig['cameras_files'][name]
                for v in vids:
                    video_loc = os.path.join(subdir, list(v.keys())[0])


                    if not os.path.exists(video_loc):
                        print(f'{subdir} does match the following structure. \n/Recording_Directory \n\t/cameraname \n\t\t/recording.mp4\n.\n.\n.')
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
    root = ttk.Window(themename=theme)
    rp = RecordingPlayer(root, "Training Set Creator", valid_videos, speed, tsconfig, training_set_config)

    save_config(rp.tsconfig, rp.config_path)

# Exports a training set created from several recordings to the format of deepethogram
def export_training_set_to_deg(recording_names, output_name, shuffle=True, split_group=True, input_behaviors=None, output_orders=None, output_behaviors=None, destination=None, scale=None, framerate=None, model_test_set=None, set='training', project_config='undefined'):
    
    """
    Export a training set to the DEG format.
    
    Parameters:
    -----------
    recording_names: list
        A list of recording names to include in the training set
    output_name: str
        The name of the output training set
    shuffle: bool
        Whether or not to shuffle the instances in the training set
    split_group: bool
        Whether or not to split the instances into separate videos
    input_behaviors: list
        A list of behaviors included in the training set
    output_orders: list 
        A list of behaviors included in the training set, ordered in the way that they should be output
    output_behaviors: list
        A list of behaviors to include in the output set
    destination: str
        The path to save the output training set
    scale: int
        The scale of the output video
    framerate: int
        The framerate of the output video
    model_test_set: str
        The name of a model to either exclude from the test set or the test set to export
    project_config: str
        The path to the project config file. If 'undefined', the function will assume that the user is located within an active project.

    """


    from cbas_headless.postprocessor import lstm_classifier

    sys.modules['lstm_classifier'] = lstm_classifier

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "lstm_classifier":
                module = "cbas_headless.postprocessor.lstm_classifier"
            return super().find_class(module, name)
    
    
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
        training_set = pconfig['trainingsets_path']
        postprocessors = pconfig['postprocessors_config']
        recordings = pconfig['recordings_path']

    
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
        training_set = pconfig['trainingsets_path']
        postprocessors = pconfig['postprocessors_config']
        recordings = pconfig['recordings_path']

    s_dataset_test = None

    recording_config = os.path.join(recordings, recording_names[0], 'details.yaml')
    if os.path.exists(recording_config):
        with open(recording_config, 'r') as file:
            rconfig = yaml.safe_load(file)
    elif scale is None or framerate is None:
        raise Exception('Scale and framerate must be defined if set is not standard.')

    # gets the test set to exclude
    if model_test_set is not None:
        with open(postprocessors, 'r') as file:
            ppconfig = yaml.safe_load(file)

        models = ppconfig['models']

        pname = None
        for name in models.keys():
            if name == model_test_set:
                pname = name 
                break
        
        if pname is None:
            raise Exception('Could not find a postprocessor with that name.')
        
        test_set = models[pname]['test_set']
        
        with open(test_set, 'rb') as file:
            s_dataset_test = CustomUnpickler(file).load()

    if destination==None:
        destination = os.path.join(training_set, output_name)
        if not os.path.exists(destination):
            os.mkdir(destination)

    if not os.path.exists(destination):
        os.mkdir(destination)

    if input_behaviors==None:
        input_behaviors = []
        for rn in recording_names:

            training_set_config = os.path.join(training_set,rn+"_training.yaml") 

            if os.path.exists(training_set_config):
                # load the preexisting test_set config
                with open(training_set_config, 'r') as file:
                    tsconfig = yaml.safe_load(file)
                
                behaviors_stored = tsconfig['behaviors']

                for b in behaviors_stored:
                    if b not in input_behaviors:
                        input_behaviors.append(b)
            else:
                raise Exception(f'No training set has been created from this {rn}.')

    if output_orders==None:
        output_orders = input_behaviors.copy()

    if output_behaviors==None:
        temp = ['background']
        for b in output_orders:
            temp.append(b)
        
        output_behaviors = temp
        output_orders = temp
    else:
        if 'background'!=output_behaviors[0]:
            temp = ['background']
            for b in output_behaviors:
                temp.append(b)
            
            output_behaviors = temp
            
            temp = ['background']
            for b in output_orders:
                temp.append(b)
            
            output_orders = temp
    
    if len(output_orders)!=len(output_behaviors):
        raise Exception('Output orders not equal in size to output behaviors.')

    for b in input_behaviors:
        if b not in output_orders:
            raise Exception('Output orders does not contain all input behaviors.')
    for b in output_orders:
        if b not in input_behaviors and b!='background':
            raise Exception('Output orders contains more behaviors than input behaviors.')


    if not split_group:

        output_file = os.path.join(destination, output_name+'.mp4')
        output_csv_file = os.path.join(destination, output_name+'_labels.csv')
        fps = framerate
        frame_size = (scale, scale)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

        labels = []

        for rn in recording_names:

            training_set_config = os.path.join(training_set,rn+"_training.yaml") 


            if os.path.exists(training_set_config):
                # load the preexisting test_set config
                with open(training_set_config, 'r') as file:
                    tsconfig = yaml.safe_load(file)
                
                behaviors_stored = tsconfig['behaviors']
                instances_stored  = tsconfig['instances']


                if scale==None:
                    scale = rconfig['scale']
                if framerate==None:
                    framerate = rconfig['framerate']




                behaviors = behaviors_stored
                instances = instances_stored

                random.shuffle(behaviors)

                unique_videos = []
                for b in behaviors:
                    
                    for inst in instances[b]:
                        if inst['video'] not in unique_videos:
                            unique_videos.append(inst['video'])

                video_dict = {vid:[] for vid in unique_videos}

                for b in behaviors:
                    for inst in instances[b]:
                        if s_dataset_test and set=='training' and inst not in s_dataset_test.instances:
                            video_dict[inst['video']].append(inst)
                        elif s_dataset_test and set=='testing' and inst in s_dataset_test.instances:
                            video_dict[inst['video']].append(inst)
                        elif model_test_set is None:
                            video_dict[inst['video']].append(inst)


                        

                total_videos = len(unique_videos)
                index = 0

                for v in range(total_videos):
                    video = unique_videos[v]
                    index+=1

                    print(f'Starting video {index}/{total_videos}...')


                    insts = video_dict[video]

                    random.shuffle(insts)
                    
                    cap = cv2.VideoCapture(video)
        
                    if not cap.isOpened():
                        print(f"Error opening video file {video}")
                        continue 
                    
                    for inst in insts:
                        start = inst['start']
                        end = inst['end']
                        label = output_behaviors[output_orders.index(inst['label'])]
                        length = inst['length']

                        label_vec = [1 if label==b else 0 for b in output_behaviors]
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

                        for f in range(0, int(length)):

        
                            # Read the next frame from the video
                            ret, frame = cap.read()
                            
                            if not ret:
                                print(f"Could not read the frame {f} from video {video}")
                            
                            # Write the frame into the output video
                            out.write(frame)

                            labels.append(label_vec)
                    
                    cap.release()

                    print(f'Finished with {index}/{total_videos} videos...')
            
            else:
                raise Exception(f'No training set has been created from this {rn}.')
                    
        out.release()


        df = pd.DataFrame(data=labels, columns=output_behaviors)
        df.to_csv(output_csv_file, index=True)

    else:


        for rn in recording_names:
            
            output_file = os.path.join(destination, output_name+'_'+rn+'.mp4')
            output_csv_file = os.path.join(destination, output_name+'_'+rn+'_labels.csv')
            fps = framerate
            frame_size = (scale, scale)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_file, fourcc, fps, frame_size)

            labels = []

            training_set_config = os.path.join(training_set,rn+"_training.yaml") 


            if os.path.exists(training_set_config):
                # load the preexisting test_set config
                with open(training_set_config, 'r') as file:
                    tsconfig = yaml.safe_load(file)
                
                behaviors_stored = tsconfig['behaviors']
                instances_stored  = tsconfig['instances']


                if scale==None:
                    scale = rconfig['scale']
                if framerate==None:
                    framerate = rconfig['framerate']




                behaviors = behaviors_stored
                instances = instances_stored

                random.shuffle(behaviors)

                unique_videos = []
                for b in behaviors:
                    
                    for inst in instances[b]:
                        if inst['video'] not in unique_videos:
                            unique_videos.append(inst['video'])

                video_dict = {vid:[] for vid in unique_videos}

                for b in behaviors:
                    for inst in instances[b]:
                        if s_dataset_test and set=='training' and inst not in s_dataset_test.instances:
                            video_dict[inst['video']].append(inst)
                        elif s_dataset_test and set=='testing' and inst in s_dataset_test.instances:
                            video_dict[inst['video']].append(inst)
                        elif model_test_set is None:
                            video_dict[inst['video']].append(inst)


                        

                total_videos = len(unique_videos)
                index = 0

                for v in range(total_videos):
                    video = unique_videos[v]
                    index+=1

                    print(f'Starting video {index}/{total_videos}...')


                    insts = video_dict[video]

                    random.shuffle(insts)
                    
                    cap = cv2.VideoCapture(video)
        
                    if not cap.isOpened():
                        print(f"Error opening video file {video}")
                        continue 
                    
                    for inst in insts:
                        start = inst['start']
                        end = inst['end']
                        label = output_behaviors[output_orders.index(inst['label'])]
                        length = inst['length']

                        label_vec = [1 if label==b else 0 for b in output_behaviors]
                        
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

                        for f in range(0, int(length)):

        
                            # Read the next frame from the video
                            ret, frame = cap.read()
                            
                            if not ret:
                                print(f"Could not read the frame {f} from video {video}")
                            
                            # Write the frame into the output video
                            out.write(frame)

                            labels.append(label_vec)
                    
                    cap.release()

                    print(f'Finished with {index}/{total_videos} videos...')
            
            else:
                raise Exception(f'No training set has been created from this {rn}.')
                    
            out.release()


            df = pd.DataFrame(data=labels, columns=output_behaviors)
            df.to_csv(output_csv_file, index=True)

