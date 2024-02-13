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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import h5py

from matplotlib import pyplot as plt

from sklearn.metrics import precision_recall_curve

import cairo

import pickle

import ttkbootstrap as ttk

from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix

import cbas_headless.postprocessor

import ttkbootstrap as ttk
theme = 'superhero'


class PieChart:
    def __init__(self, behaviors, amounts, file, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)], width=800, height=800):


        self.behaviors = behaviors
        self.amounts = amounts
        self.file = file

        self.width = width 
        self.height = height

        self.color_list = color_list


        self.draw()



    def draw(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)
        
        self.draw_piechart(ctx, self.behaviors, self.amounts, self.color_list)
        
        surface.write_to_png(self.file)



    def draw_piechart(self, ctx, behaviors, amounts, colors, LD=True):

        radius_size = .4

        cx = .5
        cy = .5

        amounts = np.array(amounts)
        amounts = amounts/np.sum(amounts)

        padding = 0
        max_rads = math.pi*2-padding*len(amounts)*math.pi*2

        padding = padding*math.pi*2

        rad1 = 0
        rad2 = 0

        for b in range(len(behaviors)):

            cl = colors[len(behaviors)-b-1]

            amount = amounts[len(behaviors)-b-1]

            ctx.move_to(cx, cy)
            ctx.arc(cx, cy, radius_size, rad1, rad1+amount*max_rads)
            ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, 1)
            ctx.fill_preserve()
            if b==len(behaviors)-1:
                ctx.set_line_width(0.005)
                ctx.set_source_rgba(0,0,0, 1)
                ctx.stroke()

                ctx.move_to(cx, cy)
                ctx.line_to(cx+radius_size, cy)
                ctx.stroke()
            else:
                ctx.set_line_width(0.005)
                ctx.set_source_rgba(0,0,0, 1)
                ctx.stroke()
                


            rad1+=amount*max_rads + padding

class PRCurves:
    def __init__(self, behaviors, trues, preds, file, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)], width=800, height=800):


        self.behaviors = behaviors
        self.trues = trues 
        self.preds = np.array(preds)
        self.file = file

        self.csv_file = os.path.splitext(self.file)[0]+'.csv'

        self.width = width 
        self.height = height

        self.color_list = color_list


        self.draw()



    def draw(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)
        
        self.draw_axes(ctx, self.behaviors, self.color_list)
        
        surface.write_to_png(self.file)





    def draw_axes(self, ctx, behaviors, colors):

        cx = .5
        cy = .5

        v_size = .9
        h_size = .9

        lx = cx - h_size/2
        ty = cy - v_size/2

        rows = int(np.sqrt(len(behaviors)))
        cols = int(np.ceil(len(behaviors)/rows))

        padding = 0.05
        
        height = (v_size - (rows-1)*padding)/rows
        width = (h_size - (cols-1)*padding)/cols


        indices = []
        total = []

        for b in range(len(behaviors)):
            row = int(b/rows) 
            col = b%rows

            ox = col*width + (col+1)*padding
            oy = (row+1)*height + (row+1)*padding


            ctx.set_line_width(0.005)
            ctx.set_source_rgba(0,0,0, 1)

            ctx.move_to(ox-.01, oy)
            ctx.line_to(ox+width, oy)
            ctx.stroke()

            ctx.move_to(ox, oy+.01)
            ctx.line_to(ox, oy-height)
            ctx.stroke()

            color = colors[b]
            label = [1 if t==b else 0 for t in self.trues]
            prob = self.preds[:, b]

            p, r = self.draw_function(ctx, ox, oy, width, height, label, prob, color)

            total.extend([p, r])

            indices.append(behaviors[b]+'_precision')
            indices.append(behaviors[b]+'_recall')
        
        df = pd.DataFrame(data=total, columns=None, index=indices)

        df.to_csv(self.csv_file, columns=None)

        

        



    def draw_function(self, ctx, ox, oy, width, height, label, prob, color):

        precision, recall, thresholds = precision_recall_curve(label, prob)

        for ind in range(0, len(recall)-1):
            ctx.set_line_width(0.005)
            ctx.set_source_rgba(color[0]/255,color[1]/255,color[2]/255, 1)

            x = recall[ind]
            y = precision[ind]

            if x<.05:
                break

            x1 = recall[ind+1]
            y1 = precision[ind+1]

            ctx.move_to(ox+width*x, oy-y*height)
            ctx.line_to(ox+width*x1, oy-y1*height)
            ctx.stroke()
        
        return (precision, recall)

class Scoreboard_ML:
    def __init__(self, parent, behaviors, pie_chart_file, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)]):
        self.parent = parent
        self.behaviors = behaviors
        self.colors = color_list
        self.pie_chart_file = pie_chart_file

        self.frame1 = tk.Frame(self.parent)
        self.frame2 = tk.Frame(self.parent)

        
        self.canvas = tk.Canvas(self.frame1, width=250, height=250)
        self.canvas.grid(column=0, row=0)

        self.frame1.grid(column=0, row=0)
        self.frame2.grid(column=1, row=0)

        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.frame2, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0)
        tk.Label(self.frame2, text="Instances", font=('Arial',9,'bold','underline')).grid(row=0, column=1)

        # Store references to the count and score variables
        self.count_vars = {}
        self.inst_vars = {}

        # Create rows for each name
        for i, name in enumerate(self.behaviors, start=1):
            tk.Label(self.frame2, text=name).grid(row=i, column=0)


            inst_var = tk.StringVar(value=str(0))  # Initial instances
            self.inst_vars[name] = inst_var
            tk.Label(self.frame2, textvariable=inst_var).grid(row=i, column=1)

    
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

    def load_image(self):
        image = Image.open(self.pie_chart_file)
        image = self.resize_image(image, 250, 250)
        img = ImageTk.PhotoImage(image)
        self.img = img
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)



    def update(self, true):

        counts = []

        for b in range(len(self.behaviors)):
            count = 0
            for inst in true:
                if inst == b:
                    count += 1
            self.inst_vars[self.behaviors[b]].set(str(count))

            counts.append(int(count))

        PieChart(self.behaviors, counts, self.pie_chart_file, color_list=self.colors)
        self.load_image()

class Metricboard_ML:
    def __init__(self, parent, behaviors, trues, preds, outputfile, sample_size=250):
        self.parent = parent
        self.behaviors = behaviors

        self.outputfile = outputfile

        self.true = trues
        self.pred = preds

        self.sample_size = sample_size
        print(self.sample_size)

        self.create_widgets()


    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0, padx=5)
        tk.Label(self.parent, text="Precision", font=('Arial',9,'bold','underline')).grid(row=0, column=2, padx=5)
        tk.Label(self.parent, text="Recall", font=('Arial',9,'bold','underline')).grid(row=0, column=3, padx=5)
        tk.Label(self.parent, text="F1-score", font=('Arial',9,'bold','underline')).grid(row=0, column=4, padx=5)
        tk.Label(self.parent, text="Balanced Accuracy", font=('Arial',9,'bold','underline')).grid(row=0, column=5, padx=5)
        tk.Label(self.parent, text="Matthew's Cor. Coef.", font=('Arial',9,'bold','underline')).grid(row=0, column=6, padx=5)

        self.precision = {}
        self.recall = {}
        self.f1_score = {}
        self.balanced_acc = {}
        self.matthews_coef = {}

        # Create rows for each name
        for i, name in enumerate(self.behaviors, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0, padx=5)


            precision_var = tk.StringVar(value="?")  # Initial precision
            self.precision[name] = precision_var
            tk.Label(self.parent, textvariable=precision_var).grid(row=i, column=2, padx=5)

            recall_var = tk.StringVar(value="?")  # Initial recall
            self.recall[name] = recall_var
            tk.Label(self.parent, textvariable=recall_var).grid(row=i, column=3, padx=5)

            f1_score_var = tk.StringVar(value="?")  # Initial f1 score
            self.f1_score[name] = f1_score_var
            tk.Label(self.parent, textvariable=f1_score_var).grid(row=i, column=4, padx=5)

            balanced_acc_var = tk.StringVar(value="?")  # Initial balanced acc
            self.balanced_acc[name] = balanced_acc_var
            tk.Label(self.parent, textvariable=balanced_acc_var).grid(row=i, column=5, padx=5)

            matthews_coef_var = tk.StringVar(value="?")  # Initial matthews coef
            self.matthews_coef[name] = matthews_coef_var
            tk.Label(self.parent, textvariable=matthews_coef_var).grid(row=i, column=6, padx=5)
        
        self.update(self.true, self.pred)


    # this is where the magic happens
    def update(self, true, pred):

        times = np.ceil(len(true)/self.sample_size)

        metrics = {b:[] for b in self.behaviors}
            
        indices = [i for i in range(len(true))]

        for t in range(int(times)):


            sample = np.random.choice(np.array(indices), size=min(self.sample_size, len(indices)))

            s_true = np.array(true)[sample]
            s_pred = np.array(pred)[sample]

            indices = [i for i in indices if i not in sample]

                
            for b in range(len(self.behaviors)):

                b_true = [l==b for l in s_true]
                b_pred = [l==b for l in s_pred]

                tp = 0
                tn = 0
                fn = 0
                fp = 0

                for i,t in enumerate(b_true):
                    if b_pred[i]==t:
                        if t==True:
                            tp+=1
                        else:
                            tn+=1
                    if b_pred[i]!=t:
                        if b_pred[i]==True:
                            fp+=1
                        else:
                            fn+=1


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
                    f1_score = 1

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

                metrics[self.behaviors[b]].append({
                    'precision':precision,
                    'recall':recall,
                    'f1_score':f1_score,
                    'balanced_acc':balanced_acc,
                    'matthews_coef':nmatthews_coef
                })

        colnames =  ['precision', 'precision_std', 'recall', 'recall_std', 'f1_score', 'f1_score_std', 'balanced_acc', 'balanced_acc_std', 'matthews_coef', 'matthews_coef_std']

        output_metrics = []

        self.outputfile1 = os.path.splitext(self.outputfile)[0]+'_raw.csv'

        all_values = []

        indnames = []

        for b in self.behaviors:

            precisions = [metric['precision'] for metric in metrics[b]]
            recalls = [metric['recall'] for metric in metrics[b]]
            f1_scores = [metric['f1_score'] for metric in metrics[b]]
            balanced_accs = [metric['balanced_acc'] for metric in metrics[b]]
            matthews_coefs = [metric['matthews_coef'] for metric in metrics[b]]

            all_values.extend([precisions, recalls, f1_scores, balanced_accs, matthews_coefs])
            indnames.extend([b+"_precisions", b+"_recalls", b+"_f1_scores", b+"_balanced_accs", b+"_matthews_coefs"])

            precision = int(np.mean(precisions)*1000)/1000
            recall = int(np.mean(recalls)*1000)/1000
            f1_score = int(np.mean(f1_scores)*1000)/1000
            balanced_acc = int(np.mean(balanced_accs)*1000)/1000
            matthews_coef = int(np.mean(matthews_coefs)*1000)/1000

            precision_std = int(np.std(precisions)*1000)/1000
            recall_std = int(np.std(recalls)*1000)/1000
            f1_score_std = int(np.std(f1_scores)*1000)/1000
            balanced_acc_std = int(np.std(balanced_accs)*1000)/1000
            matthews_coef_std = int(np.std(matthews_coefs)*1000)/1000
                    
            self.precision[b].set(value = ('%.3f +/- %.3f' % (precision, precision_std)))
            self.recall[b].set(value= ('%.3f +/- %.3f' % (recall, recall_std)))
            self.f1_score[b].set(value= ('%.3f +/- %.3f' % (f1_score, f1_score_std)))
            self.balanced_acc[b].set(value= ('%.3f +/- %.3f' % (balanced_acc, balanced_acc_std)))
            self.matthews_coef[b].set(value= ('%.3f +/- %.3f' % (matthews_coef, matthews_coef_std)))

            output_metrics.append([precision, precision_std, recall, recall_std, f1_score, f1_score_std, balanced_acc, balanced_acc_std, matthews_coef, matthews_coef_std])

        df = pd.DataFrame(data=output_metrics, columns=colnames, index=self.behaviors)
        df.to_csv(self.outputfile)

        df1 = pd.DataFrame(data=all_values, columns=None, index=indnames)
        df1.to_csv(self.outputfile1, columns=None, index=True)

class MachineLabel:
    def __init__(self, window, behaviors, trues, preds, probs, pie_chart_file, pr_curve_file, metrics_file, sample_size=250, grade=True, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)]):
        self.window = window
        self.behaviors = behaviors

        self.color_list = color_list

        self.true = trues
        self.pred = preds
        self.prob = probs
        self.pie_chart_file = pie_chart_file
        self.pr_curve_file = pr_curve_file
        self.metrics_file = metrics_file

        self.upper = tk.Frame(self.window)
        self.middle = tk.Frame(self.window)
        self.lower = tk.Frame(self.window)
        self.canvas = tk.Canvas(self.lower, width=250, height=250)
        self.canvas.grid(column=0, row=0)

        self.sb = Scoreboard_ML(self.upper, self.behaviors, self.pie_chart_file, self.color_list)
        self.sb.update(self.true)
        self.upper.pack(side='top', padx = 10, pady = 10)

        if grade:
            self.mb = Metricboard_ML(self.middle, self.behaviors, self.true, self.pred, self.metrics_file, sample_size=sample_size)
            self.mb.update(self.true, self.pred)
            self.middle.pack(side='top', padx = 10, pady = 5)
            if len(probs)!=0:
                self.pr = PRCurves(self.behaviors, self.true, self.prob, self.pr_curve_file)
                self.load_image()
                self.lower.pack(side='top', padx = 10, pady = 10)

        
        self.window.mainloop()

    
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

    def load_image(self):
        image = Image.open(self.pr_curve_file)
        image = self.resize_image(image, 400, 400)
        img = ImageTk.PhotoImage(image)
        self.img = img
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

class Scoreboard_HL:
    def __init__(self, parent, behaviors, colors, goal):
        self.parent = parent
        self.behaviors = behaviors
        self.colors = colors
        self.goal = goal
        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0)
        tk.Label(self.parent, text="Index", font=('Arial',9,'bold','underline')).grid(row=0, column=1)
        tk.Label(self.parent, text="Instances", font=('Arial',9,'bold','underline')).grid(row=0, column=2)

        # Store references to the count and score variables
        self.count_vars = {}
        self.inst_vars = {}

        # Create rows for each name
        for i, name in enumerate(self.behaviors, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0)

            # index
            tk.Label(self.parent, text=str(i), bg=self.colors[i-1]).grid(row=i, column=1, pady=3)

            inst_var = tk.StringVar(value="<"+str(self.goal))  # Initial instances
            self.inst_vars[name] = inst_var
            tk.Label(self.parent, textvariable=inst_var).grid(row=i, column=2)



    def update(self, true):

        dones = []

        for b in self.behaviors:
            count = 0
            for inst in true:
                if inst == b:
                    count += 1
            if count<self.goal:
                self.inst_vars[b].set("~"+str(int(count/10)*10)+"/"+str(self.goal))
            else:
                self.inst_vars[b].set('DONE!')
                dones.append(b)

        return dones

class Metricboard_HL:
    def __init__(self, parent, behaviors):
        self.parent = parent
        self.behaviors = behaviors

        self.true = []
        self.pred = []

        self.create_widgets()

    def create_widgets(self):
        # Create headers
        tk.Label(self.parent, text="Behavior", font=('Arial',9,'bold','underline')).grid(row=0, column=0)
        tk.Label(self.parent, text="Precision", font=('Arial',9,'bold','underline')).grid(row=0, column=2)
        tk.Label(self.parent, text="Recall", font=('Arial',9,'bold','underline')).grid(row=0, column=3)
        tk.Label(self.parent, text="F1-score", font=('Arial',9,'bold','underline')).grid(row=0, column=4)
        tk.Label(self.parent, text="Balanced Accuracy", font=('Arial',9,'bold','underline')).grid(row=0, column=5)
        tk.Label(self.parent, text="Matthew's Cor. Coef.", font=('Arial',9,'bold','underline')).grid(row=0, column=6)

        self.precision = {}
        self.recall = {}
        self.f1_score = {}
        self.balanced_acc = {}
        self.matthews_coef = {}

        # Create rows for each name
        for i, name in enumerate(self.behaviors, start=1):
            tk.Label(self.parent, text=name).grid(row=i, column=0)


            precision_var = tk.StringVar(value="?")  # Initial precision
            self.precision[name] = precision_var
            tk.Label(self.parent, textvariable=precision_var).grid(row=i, column=2)

            recall_var = tk.StringVar(value="?")  # Initial recall
            self.recall[name] = recall_var
            tk.Label(self.parent, textvariable=recall_var).grid(row=i, column=3)

            f1_score_var = tk.StringVar(value="?")  # Initial f1 score
            self.f1_score[name] = f1_score_var
            tk.Label(self.parent, textvariable=f1_score_var).grid(row=i, column=4)

            balanced_acc_var = tk.StringVar(value="?")  # Initial balanced acc
            self.balanced_acc[name] = balanced_acc_var
            tk.Label(self.parent, textvariable=balanced_acc_var).grid(row=i, column=5)

            matthews_coef_var = tk.StringVar(value="?")  # Initial matthews coef
            self.matthews_coef[name] = matthews_coef_var
            tk.Label(self.parent, textvariable=matthews_coef_var).grid(row=i, column=6)


    # this is where the magic happens
    def update(self, true, pred):


        metrics = {b:[] for b in self.behaviors}
            
        for b in self.behaviors:

            b_true = [l==b for l in true]
            b_pred = [l==b for l in pred]

            tp = 0
            tn = 0
            fn = 0
            fp = 0

            for i,t in enumerate(b_true):
                if b_pred[i]==t:
                    if t==True:
                        tp+=1
                    else:
                        tn+=1
                if b_pred[i]!=t:
                    if b_pred[i]==True:
                        fp+=1
                    else:
                        fn+=1


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
                f1_score = 1

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


        for b in self.behaviors:

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

class HumanLabel:
    def __init__(self, window, window_title, instance_paths, testing_config, behaviors_ordered=None, seq_len=7, grade=True, replays=3, goal=2, trues=None, preds=None):
        self.window = window
        self.window.title(window_title)
        self.instances = []
        self.testing_config = testing_config

        instances = []
        test_instances = []

        self.behaviors = []
        if behaviors_ordered is not None:
            self.behaviors = behaviors_ordered


        self.cur_inst_label = None 
        self.trues = []
        self.preds = []


        if trues is not None:
            self.trues.extend(trues)
        if preds is not None:
            self.preds.extend(preds)

        self.seq_len = seq_len
        self.goal = goal

        self.frame_ind = 0
        self.replays = replays

        for instance_path in instance_paths:
            #print(instance_path)
            with open(instance_path) as file:
                training_set = yaml.safe_load(file)

                behaviors = training_set['behaviors']
                for b in behaviors:
                    if b not in self.behaviors:
                        if behaviors_ordered is None:
                            self.behaviors.append(b)
                    for inst in training_set['instances'][b]:
                            instances.append(inst)


        self.instances = instances
        self.test_instances = test_instances

        self.upper = tk.Frame(self.window)
        

        self.color_list = ['coral','goldenrod1','royalblue','red1','deeppink1','lightslateblue','gray23','limegreen','cornsilk']


        # Load video and predictions
        self.cap = cv2.VideoCapture(self.instances[0]['video'])
        

        self.frame_num = 0

        # Create a canvas that can fit the above video source size
        self.canvas = tk.Canvas(self.upper, width=self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.grid(column=0, row=0)

        self.board_frame = tk.Frame(self.upper)
        self.scoreboard = Scoreboard_HL(self.board_frame, self.behaviors, self.color_list, self.goal)
        self.scoreboard.update(self.trues)
        self.board_frame.grid(column=1, row=0)


        self.upper.pack(anchor=tk.CENTER, pady=5, padx=5)

        
        self.window.bind('r', lambda i: self.replay())

        for i in range(1,len(self.behaviors)+1):
            self.window.bind(str(i), lambda x, i=i: self.guess(i - 1))


        self.metricboard_frame = tk.Frame(self.window)
        self.metricboard = None

        if grade==True:
            self.metricboard = Metricboard_HL(self.metricboard_frame, self.behaviors)
            #self.metricboard.update(self.trues, self.preds)

        self.metricboard_frame.pack(anchor=tk.CENTER, pady=5, padx=5)

        self.frames = []


        # Update & delay variables
        self.delay = 50   # ms

        self.next()
        self.update()

        self.window.mainloop()
    
    def update(self):

        self.window.after(self.delay, self.update)
    
    def replay(self):
        self.frame_ind = 0
        self.play_video()
    
    def guess(self, i):

        pred = self.behaviors[i]
        self.trues.append(self.cur_inst_label)
        self.preds.append(pred)

        dones = self.scoreboard.update(self.trues)

        if set(dones)==set(self.behaviors):
            self.metricboard.update(self.trues, self.preds)

        self.next()

    def play_video(self):
        # Read the next frame from the video. If reached the end of the video, release the video capture object
        ret, frame = self.frames[self.frame_ind%len(self.frames)]
        if self.frame_ind>(self.seq_len*2+1)*self.replays:
            return
        if ret:
            self.canvas.delete("all")
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.frame_ind+=1
            self.window.after(self.delay, self.play_video)
           

    def next(self):

        # clear the last frames 
        self.frames = []
        self.frame_ind = 0


        # randomly select an instance
        iidx = random.randint(0, len(self.instances)-1)
        inst = self.instances[iidx]

        label = inst['label']
        video = inst['video']
        start = inst['start']
        end = inst['end']
        self.cur_inst_label = label

        self.cap = cv2.VideoCapture(video)

        # Check if the video file was opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open video file.")
            exit()

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # randomly select a position in the range of the instance
        central_frame = random.randint(start, end-1)
        start_frame = central_frame - self.seq_len 
        num_frames = self.seq_len*2+1

        if start_frame<1:
            num_frames = num_frames+(1-start_frame)
            start_frame = 1
        
        if start_frame+num_frames > total_frames:
            num_frames += total_frames-1-(start_frame+num_frames)

        # extract those frames from the video

        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)  # Subtract 1 because frame numbering starts from 0

        for i in range(num_frames):
            # Read the frame
            ret, frame = self.cap.read()

            # Check if the frame was read successfully
            if ret:
                self.frames.append((ret, frame))
            else:
                print(f"Error: Unable to extract frame {i+start_frame}")

        # Release the video capture object
        self.cap.release()


        self.play_video()

   
# This function is just expecting a directory with subdirectories containing mp4 videos and labels in the DEG form
def human_validation(instance_paths, behaviors=[], grader="Anonymous", project_config='undefined'):

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

        # grabbing the location of the human label dir
        human_label = pconfig['humanlabel_path']

    
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

        # grabbing the location of the human label dir
        human_label = pconfig['humanlabel_path']


    hl_trues = os.path.join(human_label,grader+"_true.npy")
    hl_preds = os.path.join(human_label,grader+"_pred.npy")

    trues=None 
    preds=None

    if os.path.exists(hl_trues):
        with open(hl_trues, 'rb') as f:
            trues = np.load(f)
    if os.path.exists(hl_preds):
        with open(hl_preds, 'rb') as f:
            preds = np.load(f)

    root = tk.Tk()
    hl = HumanLabel(root, 'Human Labeling', instance_paths, testing_config=None, behaviors_ordered=behaviors, goal=1000, trues=trues, preds=preds)

    with open(hl_trues, 'wb') as f:
        np.save(f, np.array(hl.trues))
    with open(hl_preds, 'wb') as f:
        np.save(f, np.array(hl.preds))

def postprocessor_validation(postprocessor_name, test_samples, project_config='undefined'):

    
    from cbas_headless.postprocessor import lstm_classifier
    from cbas_headless.postprocessor.lstm_classifier import S_Features_Test
    from cbas_headless.postprocessor.lstm_classifier import Encoder
    from cbas_headless.postprocessor.lstm_classifier import Decoder
    from cbas_headless.postprocessor.lstm_classifier import Classifier

    sys.modules['lstm_classifier'] = lstm_classifier

    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "lstm_classifier":
                module = "cbas_headless.postprocessor.lstm_classifier"
            return super().find_class(module, name)



    # open the project config and get the postprocessor yaml path
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']

    
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']
    
    
    with open(postprocessors, 'r') as file:
        ppconfig = yaml.safe_load(file)

    models = ppconfig['models']


    pname = None
    for name in models.keys():
        if name == postprocessor_name:
            pname = name 
            break
    
    if pname is None:
        raise Exception('Could not find a postprocessor with that name.')
    
    test_set = models[pname]['test_set']
    
    with open(test_set, 'rb') as file:
        s_dataset_test = CustomUnpickler(file).load()


    test_set = DataLoader(s_dataset_test, batch_size=128, shuffle=True, num_workers=8)

    m_path = models[pname]['classifier']

    behaviors = models[pname]['behaviors']

    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the state dict from the old model
    classifier = torch.load(m_path, map_location=device)


    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Move your model to the device (GPU if available)
    classifier.to(device)

    classifier.eval()

        
    # Initialize empty lists to store predicted and true labels
    all_preds = []
    all_probs = []
    all_trues = []

    for seq, target in test_set:

        # Move data to the device
        target = target.long()
        target = torch.argmax(target, dim=1)
        seq, target = seq.to(device), target.to(device)
        output = classifier(seq)

        # Convert predicted and target values to integers
        pred_indices = output.argmax(dim=1).detach().cpu().numpy()
        probs = F.softmax(output).detach().cpu().numpy()
        true_indices = target.detach().cpu().numpy()

        # Append indices to the lists
        all_preds.extend(pred_indices)
        all_probs.extend(probs)
        all_trues.extend(true_indices)


    pie_chart_path = os.path.join(figures, postprocessor_name+'_testset.png')
    
    pr_curve_file = os.path.join(figures, postprocessor_name+'_prcurve.png')

    metrics_file = os.path.join(figures, postprocessor_name+'_metrics.csv')
    
    root = ttk.Window(themename=theme)
    ml = MachineLabel(root, behaviors, trues = all_trues, preds = all_preds, probs = all_probs, pie_chart_file=pie_chart_path, pr_curve_file=pr_curve_file, metrics_file=metrics_file, sample_size=int(len(all_trues)/test_samples))

def deg_cross_validation(postprocessor_name, degname, test_samples, project_config='undefined'):

    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "lstm_classifier":
                module = "cbas_headless.postprocessor.lstm_classifier"
            return super().find_class(module, name)


    # open the project config and get the postprocessor yaml path
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']

    
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']
    
    
    with open(postprocessors, 'r') as file:
        ppconfig = yaml.safe_load(file)

    models = ppconfig['models']


    pname = None
    for name in models.keys():
        if name == postprocessor_name:
            pname = name 
            break
    
    if pname is None:
        raise Exception('Could not find a postprocessor with that name.')
    
    test_set = models[pname]['test_set']
    
    with open(test_set, 'rb') as file:
        s_dataset_test = CustomUnpickler(file).load()

    behaviors = models[pname]['behaviors']

    
        
    # Initialize empty lists to store predicted and true labels
    all_preds = []
    all_probs = []
    all_trues = []

    predictions_dfs = {instance['video']: os.path.splitext(instance['video'])[0] + '_outputs.h5' for instance in s_dataset_test.instances}

    for i in range(10000):
        
        iidx = i%len(s_dataset_test.instances)
        instance = s_dataset_test.instances[iidx]

        label = behaviors.index(instance['label'])
        start = instance['start'] - 1

        df_file = predictions_dfs[instance['video']]

        with h5py.File(df_file, 'r') as f:
            df = np.array(f['tgmj']['P'])


        frame = int(start+random.randint(0,int(instance['length']-1)))

        counts = [0 for b in behaviors]

        ind = np.argmax(df[frame,:])-1
        if ind<0:
            ind = len(behaviors)-1
        counts[ind] += 1
        
        pred = np.argmax(counts)

        prob = df[frame,1:].tolist()
        prob.append(df[frame,0])
        prob = np.array(prob)


        all_trues.append(label)
        all_preds.append(pred)
        all_probs.append(prob)
        


    pie_chart_path = os.path.join(figures, degname+'_testset.png')
    
    pr_curve_file = os.path.join(figures, degname+'_prcurve.png')

    metrics_file = os.path.join(figures, degname+'_metrics.csv')
    
    root = tk.Tk()
    ml = MachineLabel(root, behaviors, trues = all_trues, preds = all_preds, probs = all_probs, pie_chart_file=pie_chart_path, pr_curve_file=pr_curve_file, metrics_file=metrics_file, sample_size=int(len(all_trues)/test_samples))

def display_set(postprocessor_name, settype='test', frames=False, project_config='undefined'):

    from cbas_headless.postprocessor import lstm_classifier
    from cbas_headless.postprocessor.lstm_classifier import S_Features
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "lstm_classifier":
                module = "cbas_headless.postprocessor.lstm_classifier"
            return super().find_class(module, name)


    # open the project config and get the postprocessor yaml path
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']

    
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']
    
    
    with open(postprocessors, 'r') as file:
        ppconfig = yaml.safe_load(file)

    models = ppconfig['models']


    pname = None
    for name in models.keys():
        if name == postprocessor_name:
            pname = name 
            break
    
    if pname is None:
        raise Exception('Could not find a postprocessor with that name.')

    if settype=='test':
        test_set = models[pname]['test_set']
        
        with open(test_set, 'rb') as file:
            dataset = CustomUnpickler(file).load()
    elif settype=='training':
        training_set = models[pname]['training_set']
        
        with open(training_set, 'rb') as file:
            dataset = CustomUnpickler(file).load()
    else:
        raise Exception("Set type must be either 'test' or 'training'.")
    

    behaviors = models[pname]['behaviors']

    
    # Initialize empty lists to store predicted and true labels
    all_trues = []
    all_probs = []
    all_preds = []

    for i in range(len(dataset.instances)):
        
        iidx = i%len(dataset.instances)
        instance = dataset.instances[iidx]

        label = behaviors.index(instance['label'])

        if frames:
            for j in range(int(instance['length'])):
                all_trues.append(label)

        else:
            all_trues.append(label)

    
    descr = 'instances'
        
    if frames:
        descr = 'frames'

    pie_chart_path = os.path.join(figures, postprocessor_name+'_'+settype+'_set_'+descr+'.png')
    pr_curve_file = os.path.join(figures, postprocessor_name+'_prcurve.png')
    metrics_file = os.path.join(figures, postprocessor_name+'_metrics.csv')
        
    
    root = tk.Tk()
    ml = MachineLabel(root, behaviors, trues = all_trues, preds = all_preds, probs = all_probs, pie_chart_file=pie_chart_path, pr_curve_file=pr_curve_file, metrics_file=metrics_file, grade=False)

def calc_avg_bout_len(postprocessor_name, settype='training', frames=False, project_config='undefined'):

    from cbas_headless.postprocessor import lstm_classifier
    from cbas_headless.postprocessor.lstm_classifier import S_Features
    
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "lstm_classifier":
                module = "cbas_headless.postprocessor.lstm_classifier"
            return super().find_class(module, name)


    # open the project config and get the postprocessor yaml path
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']

    
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

        # grabbing the locations of the postprocessors
        postprocessors = pconfig['postprocessors_config']
        figures = pconfig['figures_path']
    
    
    with open(postprocessors, 'r') as file:
        ppconfig = yaml.safe_load(file)

    models = ppconfig['models']


    pname = None
    for name in models.keys():
        if name == postprocessor_name:
            pname = name 
            break
    
    if pname is None:
        raise Exception('Could not find a postprocessor with that name.')

    if settype=='test':
        test_set = models[pname]['test_set']
        
        with open(test_set, 'rb') as file:
            dataset = CustomUnpickler(file).load()
    elif settype=='training':
        training_set = models[pname]['training_set']
        
        with open(training_set, 'rb') as file:
            dataset = CustomUnpickler(file).load()
    else:
        raise Exception("Set type must be either 'test' or 'training'.")
    

    behaviors = models[pname]['behaviors']

    behavior_bouts = {b:[] for b in behaviors}


    for i in range(len(dataset.instances)):
        
        iidx = i%len(dataset.instances)
        instance = dataset.instances[iidx]

        label = instance['label']

        length = instance['length']

        behavior_bouts[label].append(length)

    m = 0
    for b in behaviors:
        if len(behavior_bouts[b]) > m:
            m = len(behavior_bouts[b])

    for b in behaviors:
        print('Average bout length of %s is %.2f' % (b, np.mean(behavior_bouts[b])))

    for b in behaviors:
        print('Most probable bout length of %s is %.2f' % (b, np.mean(np.multiply(behavior_bouts[b], behavior_bouts[b]))))

    for b in behaviors:
        for e in range(0, (len(behavior_bouts[b])-m)*-1):
            behavior_bouts[b].append(0)


    
    raw_bouts_path = os.path.join(figures, postprocessor_name+'_bouts_raw.csv')

    df = pd.DataFrame.from_dict(data=behavior_bouts, orient='columns')

    df.to_csv(raw_bouts_path)
    
def human_validation(grader, behaviors, test_samples, project_config='undefined'):



    # open the project config
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

        figures = pconfig['figures_path']
        hl_dir = pconfig['humanlabel_path']

    
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

        figures = pconfig['figures_path']
        hl_dir = pconfig['humanlabel_path']
        
    


    hl_trues = os.path.join(hl_dir,grader+"_true.npy")
    hl_preds = os.path.join(hl_dir,grader+"_pred.npy")

    trues=[] 
    preds=[]
    probs=[]

    if os.path.exists(hl_trues):
        with open(hl_trues, 'rb') as f:
            trues = np.load(f)
    if os.path.exists(hl_preds):
        with open(hl_preds, 'rb') as f:
            preds = np.load(f)


    trues = [behaviors.index(b) for b in trues]
    preds = [behaviors.index(b) for b in preds]





    pie_chart_path = os.path.join(figures, grader+'_testset.png')
    
    pr_curve_file = os.path.join(figures, grader+'_prcurve.png')

    metrics_file = os.path.join(figures, grader+'_metrics.csv')
    
    root = tk.Tk()
    ml = MachineLabel(root, behaviors, trues = trues, preds = preds, probs = probs, pie_chart_file=pie_chart_path, pr_curve_file=pr_curve_file, metrics_file=metrics_file, sample_size=int(len(trues)/test_samples))

def output_example_images(recording_name, output_folder, counts, length=10, project_config='undefined'):
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

        # grabbing the locations of the training sets
        training_sets = pconfig['trainingsets_path']

    
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

        # grabbing the locations of the training sets
        training_sets = pconfig['trainingsets_path']

    training_set_config = os.path.join(training_sets,recording_name+"_training.yaml") 

    if os.path.exists(training_set_config):
        # load the preexisting test_set config
        with open(training_set_config, 'r') as file:
            tsconfig = yaml.safe_load(file)
        
        behaviors_stored = tsconfig['behaviors']
        instances_stored  = tsconfig['instances']

        behaviors = behaviors_stored
        instances = instances_stored

        for b in behaviors:
            folder = os.path.join(output_folder, b)
            if not os.path.exists(folder):
                os.mkdir(folder)

            beh_insts = instances[b]
            random.shuffle(beh_insts)

            for c in range(counts):
                subfolder = os.path.join(folder, 'example_'+str(c))
                if not os.path.exists(subfolder):
                    os.mkdir(subfolder)
                if c<len(beh_insts):
                    inst = beh_insts[c]
                    video = inst['video']
                    start = inst['start']

                    cap = cv2.VideoCapture(video)

                    frame_number = int(start)

                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                        for l in range(length):
                            ret, frame = cap.read()
                            if ret:
                                # Define the filename of the frame
                                frame_filename = os.path.join(subfolder, f'frame_{l:04d}.png')
                                
                                # Save the frame as a PNG file
                                cv2.imwrite(frame_filename, frame)
                                
                            else:
                                break
    
    else:
        raise Exception('No such training set exists.')

