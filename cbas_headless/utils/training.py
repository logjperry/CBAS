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

from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix

class RecordingPlayer:
    def __init__(self, window, window_title, video_paths, speed, tsconfig, config_path, grade=False):
        self.window = window
        self.window.title(window_title)
        self.tsconfig = tsconfig
        self.video_paths = video_paths
        self.config_path = config_path
        self.instance_stack = []

        self.instance_cache = []
        self.video_index = 0

        self.upper = tk.Frame(self.window)


        self.prediction_paths = {}
        self.probability_paths = {}

        if grade==True:
            for v in video_paths:
                pred = os.path.splitext(v)[0]+'_predictions.csv'
                if os.path.exists(pred):
                    self.prediction_paths[v] = pred
                probs = os.path.splitext(v)[0]+'_probs.csv'
                if os.path.exists(probs):
                    self.probability_paths[v] = probs

            
        
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
        self.canvas = tk.Canvas(self.upper, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(column=0, row=0)

        self.start = -1 
        self.end = -1

        # self.startButton = tk.Button(window, text='Start', command=self.start_frame).pack(anchor=tk.CENTER)
        # self.endButton = tk.Button(window, text='End', command=self.end_frame).pack(anchor=tk.CENTER)

        

        
        self.board_frame = tk.Frame(self.upper)
        self.scoreboard = Scoreboard(self.board_frame, self.behaviors, self.color_list)
        self.board_frame.grid(column=1, row=0)
    
        self.ss = tk.StringVar(value='Surf Speed = '+str(self.speed))
        self.speed_label = tk.Label(self.upper, textvariable=self.ss, pady=5)
        self.speed_label.grid(column=0, row=1)

        self.upper.pack(anchor=tk.CENTER, pady=5, padx=5)


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

        self.metricboard_frame = tk.Frame(self.window)
        self.metricboard = None

        if grade==True:
            self.metricboard = Metricboard(self.metricboard_frame, self.behaviors, self.prediction_paths, self.probability_paths)
            self.metricboard.update(self.tsconfig)

        self.metricboard_frame.pack(anchor=tk.CENTER, pady=5, padx=5)


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
        tk.Label(self.parent, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0)
        tk.Label(self.parent, text="Index", font=('Arial',9,'bold','underline')).grid(row=0, column=1)
        tk.Label(self.parent, text="Frames", font=('Arial',9,'bold','underline')).grid(row=0, column=2)
        tk.Label(self.parent, text="Instances", font=('Arial',9,'bold','underline')).grid(row=0, column=3)

        # Store references to the count and score variables
        self.count_vars = {}
        self.inst_vars = {}

        # Create rows for each name
        for i, name in enumerate(self.names, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0)

            # index
            tk.Label(self.parent, text=str(i), bg=self.colors[i-1]).grid(row=i, column=1, pady=3)

            count_var = tk.StringVar(value="0")  # Initial count
            self.count_vars[name] = count_var
            tk.Label(self.parent, textvariable=count_var).grid(row=i, column=2)

            inst_var = tk.StringVar(value="0")  # Initial instances
            self.inst_vars[name] = inst_var
            tk.Label(self.parent, textvariable=inst_var).grid(row=i, column=3)



    def update(self, config):
        behaviors = config['behaviors']
        instances = config['instances']

        for b in behaviors:
            count = 0
            for inst in instances[b]:
                count += inst['length']
            self.count_vars[b].set(str(count))
            self.inst_vars[b].set(str(len(instances[b])))

class Metricboard:
    def __init__(self, parent, names, prediction_paths=None, probability_paths=None):
        self.parent = parent
        self.names = names
        self.prediction_paths = prediction_paths
        self.probability_paths = probability_paths

        self.maximize = False
        if self.probability_paths!=None:
            self.maximize = True

        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0)
        tk.Label(self.parent, text="Precision", font=('Arial',9,'bold','underline')).grid(row=0, column=1)
        tk.Label(self.parent, text="Recall", font=('Arial',9,'bold','underline')).grid(row=0, column=2)
        tk.Label(self.parent, text="F1-score", font=('Arial',9,'bold','underline')).grid(row=0, column=3)
        tk.Label(self.parent, text="Balanced Accuracy", font=('Arial',9,'bold','underline')).grid(row=0, column=4)
        tk.Label(self.parent, text="Matthew's Cor. Coef.", font=('Arial',9,'bold','underline')).grid(row=0, column=5)

        self.precision = {}
        self.recall = {}
        self.f1_score = {}
        self.balanced_acc = {}
        self.matthews_coef = {}

        # Create rows for each name
        for i, name in enumerate(self.names, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0)


            precision_var = tk.StringVar(value="1")  # Initial precision
            self.precision[name] = precision_var
            tk.Label(self.parent, textvariable=precision_var).grid(row=i, column=1)

            recall_var = tk.StringVar(value="1")  # Initial recall
            self.recall[name] = recall_var
            tk.Label(self.parent, textvariable=recall_var).grid(row=i, column=2)

            f1_score_var = tk.StringVar(value="1")  # Initial f1 score
            self.f1_score[name] = f1_score_var
            tk.Label(self.parent, textvariable=f1_score_var).grid(row=i, column=3)

            balanced_acc_var = tk.StringVar(value="1")  # Initial balanced acc
            self.balanced_acc[name] = balanced_acc_var
            tk.Label(self.parent, textvariable=balanced_acc_var).grid(row=i, column=4)

            matthews_coef_var = tk.StringVar(value="1")  # Initial matthews coef
            self.matthews_coef[name] = matthews_coef_var
            tk.Label(self.parent, textvariable=matthews_coef_var).grid(row=i, column=5)

    def update(self, config):
        if self.maximize:
            self.update_max(config)
        else:
            self.update_nomax(config)


    # this is where the magic happens
    def update_nomax(self, config):
        behaviors = config['behaviors']
        instances = config['instances']

        counts = []

        loaded_dfs = {}

        for b in behaviors:
            count = 0
            for inst in instances[b]:
                count += inst['length']
            counts.append(count)
    
        counts = np.array(counts)
        minimum = counts.min()

        indices = {b:[i for i in range(0, len(instances[b]))] for b in behaviors}

        metrics = {b:[] for b in behaviors}

        shuffles = 10

        for i in range(0, shuffles):

            matrix = {b:{'tp':0,'tn':0,'fp':0,'fn':0} for b in behaviors}

            for b in behaviors:
                count = 0

                # create some non-repeating random indexes
                # add those instances until the count reaches the minimum
                indexes = [i for i in range(0, len(instances[b]))]
                random.shuffle(indexes)

                real_instances = []

                i = 0
                while count<minimum and i<len(indexes):
                    inst = instances[b][int(indexes[i])]

                    # load the df
                    if inst['video'] not in loaded_dfs.keys():
                        if inst['video'] not in self.prediction_paths.keys():
                            continue 
                        else:
                            loaded_dfs[inst['video']] = pd.read_csv(self.prediction_paths[inst['video']])

                    count += inst['length']
                    real_instances.append(inst)

                    i+=1
                

                
                for inst in real_instances:
                    # get the prediction counts for each instance 
                    start = inst['start']
                    end = inst['end']

                    length = inst['length']

                    label = inst['label']

                    df = loaded_dfs[inst['video']]
                    headers = df.columns.to_numpy()
                    headers = headers[1:]

                    predictions = df.iloc[int(start):int(end)].to_numpy()
                    predictions = predictions[:,1:]

                    values = {b:0 for b in behaviors}

                    for r in predictions:
                        labels = headers[r==1]
                        for l in labels:
                            if l == 'background':
                                continue
                            values[l]+=1
                    
                    for k in values.keys():
                        if k==label:
                            matrix[label]['tp']+=values[k]
                        else:
                            matrix[k]['fp']+=values[k]
                            matrix[label]['fn']+=values[k]

                        if k!=label:
                            matrix[k]['tn']+=length-values[k]
            
            for b in behaviors:

                tp = matrix[b]['tp']
                tn = matrix[b]['tn']
                fp = matrix[b]['fp']
                fn = matrix[b]['fn']

                if tp+fp!=0: 
                    precision = tp/(tp+fp)
                    precision = int(precision*1000)/1000
                else:
                    precision = 1

                if tp+fn!=0:
                    recall = tp/(tp+fn)
                    recall = int(recall*1000)/1000
                else:
                    recall = 1

                if precision+recall!=0:
                    f1_score = 2*precision*recall/(precision+recall)
                    f1_score = int(f1_score*1000)/1000
                else:
                    f1_score = 0

                if tn+fp!=0:
                    specificity = tn/(tn+fp)
                    specificity = int(specificity*1000)/1000
                else:
                    specificity = 1

                balanced_acc = (recall+specificity)/2
                balanced_acc = int(balanced_acc*1000)/1000

                if np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))!=0:
                    matthews_coef = (tn*tp - fn*fp)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
                    nmatthews_coef = (matthews_coef+1)/2
                    nmatthews_coef = int(nmatthews_coef*1000)/1000
                else:
                    nmatthews_coef = 1

                metrics[b].append({
                    'precision':precision,
                    'recall':recall,
                    'f1_score':f1_score,
                    'balanced_acc':balanced_acc,
                    'matthews_coef':nmatthews_coef
                })


        for b in behaviors:

            precisions = [metric['precision'] for metric in metrics[b]]
            recalls = [metric['recall'] for metric in metrics[b]]
            f1_scores = [metric['f1_score'] for metric in metrics[b]]
            balanced_accs = [metric['balanced_acc'] for metric in metrics[b]]
            matthews_coefs = [metric['matthews_coef'] for metric in metrics[b]]

            precision = int(np.mean(precisions)*1000)/1000
            recall = int(np.mean(recalls)*1000)/1000
            f1_score = int(np.mean(f1_scores)*1000)/1000
            balanced_acc = int(np.mean(balanced_accs)*1000)/1000
            matthews_coef = int(np.mean(matthews_coefs)*1000)/1000

                    
            self.precision[b].set(value=str(precision))
            self.recall[b].set(value=str(recall))
            self.f1_score[b].set(value=str(f1_score))
            self.balanced_acc[b].set(value=str(balanced_acc))
            self.matthews_coef[b].set(value=str(matthews_coef))



    def update_max(self, config):
        behaviors = config['behaviors']
        instances = config['instances']

        counts = []

        loaded_dfs = {}

        for b in behaviors:
            count = 0
            for inst in instances[b]:
                count += inst['length']
            counts.append(count)
    
        counts = np.array(counts)
        minimum = counts.min()

        indices = {b:[i for i in range(0, len(instances[b]))] for b in behaviors}

        metrics = {b:[] for b in behaviors}

        shuffles = 100

        for i in range(0, shuffles):


            matrix = {b:{'tp':0,'tn':0,'fp':0,'fn':0} for b in behaviors}

            labels_list = []
            preds_list = []
            for b in behaviors:
                count = 0

                # create some non-repeating random indexes
                # add those instances until the count reaches the minimum
                indexes = [i for i in range(0, len(instances[b]))]
                random.shuffle(indexes)

                real_instances = []

                i = 0
                while count<minimum and i<len(indexes):
                    inst = instances[b][int(indexes[i])]

                    # load the df
                    if inst['video'] not in loaded_dfs.keys():
                        if inst['video'] not in self.probability_paths.keys():
                            continue 
                        else:
                            loaded_dfs[inst['video']] = pd.read_csv(self.probability_paths[inst['video']])

                    count += inst['length']
                    real_instances.append(inst)

                    i+=1
                
                
                for inst in real_instances:
                    # get the prediction counts for each instance 
                    start = inst['start']
                    end = inst['end']

                    length = inst['length']

                    label = inst['label']

                    df = loaded_dfs[inst['video']]
                    headers = df.columns.to_numpy()
                    headers = headers[2:]

                    label_row = np.where(headers == label, 1, 0)

                    predictions = df.iloc[int(start):int(end)].to_numpy()
                    predictions = predictions[:,2:]


                    for i in range(0,int(length)):
                        labels_list.append(label_row)

                    preds_list.extend(predictions)
                
            preds_list = np.array(preds_list)
            labels_list = np.array(labels_list)

            thresholds = {b:.5 for b in behaviors}

            i = 1
            for b in behaviors: # For each class
                best_f1 = 0
                best_threshold = 0.5
                for threshold in np.linspace(0, 1, 100): # You can increase 100 to be more precise
                    # Create binary predictions for this class
                    y_pred = preds_list[:, i] >= threshold
                    y_pred = y_pred.astype(int)
                    
                    # Convert multi-class labels to binary
                    y_true_binary = labels_list[:, i]

                    # Calculate F1 score
                    f1_s = f1(y_true_binary, y_pred, zero_division=0)

                    if f1_s > best_f1:
                        best_f1 = f1_s
                        best_threshold = threshold


                thresholds[b] = best_threshold
                i+=1

            print(thresholds)

            i = 1
            for b in behaviors:
                threshold = thresholds[b]

                tn, fp, fn, tp = confusion_matrix(labels_list[:,i], preds_list[:,i]>threshold).ravel()
                matrix[b]['tp'] = tp
                matrix[b]['tn'] = tn
                matrix[b]['fp'] = fp
                matrix[b]['fn'] = fn
                                       
                i+=1
                    
                    
            
            for b in behaviors:

                tp = matrix[b]['tp']
                tn = matrix[b]['tn']
                fp = matrix[b]['fp']
                fn = matrix[b]['fn']

                if tp+fp!=0: 
                    precision = tp/(tp+fp)
                    precision = int(precision*1000)/1000
                else:
                    precision = 1

                if tp+fn!=0:
                    recall = tp/(tp+fn)
                    recall = int(recall*1000)/1000
                else:
                    recall = 1

                if precision+recall!=0:
                    f1_score = 2*precision*recall/(precision+recall)
                    f1_score = int(f1_score*1000)/1000
                else:
                    f1_score = 0

                if tn+fp!=0:
                    specificity = tn/(tn+fp)
                    specificity = int(specificity*1000)/1000
                else:
                    specificity = 1

                balanced_acc = (recall+specificity)/2
                balanced_acc = int(balanced_acc*1000)/1000

                if np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))!=0:
                    matthews_coef = (tn*tp - fn*fp)/(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
                    nmatthews_coef = (matthews_coef+1)/2
                    nmatthews_coef = int(nmatthews_coef*1000)/1000
                else:
                    nmatthews_coef = 1

                metrics[b].append({
                    'precision':precision,
                    'recall':recall,
                    'f1_score':f1_score,
                    'balanced_acc':balanced_acc,
                    'matthews_coef':nmatthews_coef
                })


        for b in behaviors:

            precisions = [metric['precision'] for metric in metrics[b]]
            recalls = [metric['recall'] for metric in metrics[b]]
            f1_scores = [metric['f1_score'] for metric in metrics[b]]
            balanced_accs = [metric['balanced_acc'] for metric in metrics[b]]
            matthews_coefs = [metric['matthews_coef'] for metric in metrics[b]]

            precision = int(np.mean(precisions)*1000)/1000
            recall = int(np.mean(recalls)*1000)/1000
            f1_score = int(np.mean(f1_scores)*1000)/1000
            balanced_acc = int(np.mean(balanced_accs)*1000)/1000
            matthews_coef = int(np.mean(matthews_coefs)*1000)/1000

                    
            self.precision[b].set(value=str(precision))
            self.recall[b].set(value=str(recall))
            self.f1_score[b].set(value=str(f1_score))
            self.balanced_acc[b].set(value=str(balanced_acc))
            self.matthews_coef[b].set(value=str(matthews_coef))
            





        


def save_config(tsconfig, config_path):
    with open(config_path, 'w+') as file:
        yaml.dump(tsconfig, file, allow_unicode=True)


# This function is just expecting a directory with subdirectories containing mp4 videos and labels in the DEG form
def create_training_set(recording_path, behaviors=[], project_config='undefined', grade=False):

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
    rp = RecordingPlayer(root, "Training Set Creator", valid_videos, speed, tsconfig, training_set_config, grade=grade)

    save_config(rp.tsconfig, rp.config_path)

