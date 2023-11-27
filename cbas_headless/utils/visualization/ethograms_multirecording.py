
import os
import glob
import time
import traceback
import subprocess
import threading
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
from queue import Queue
from typing import Union
import sys
from omegaconf import DictConfig, OmegaConf
import math

from datetime import datetime

import numpy as np
import tkinter as tk
import matplotlib as mpl
from matplotlib.patches import Rectangle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

import numpy as np

import numpy as np
from scipy.optimize import least_squares







class EthogramWindow(NavigationToolbar2Tk):
    def __init__(self,canvas_,parent_):
        self.toolitems = (
            ('Home', 'Lorem ipsum dolor sit amet', 'home', 'home'),
            ('Back', 'consectetuer adipiscing elit', 'back', 'back'),
            ('Forward', 'sed diam nonummy nibh euismod', 'forward', 'forward'),
            (None, None, None, None),
            ('Pan', 'tincidunt ut laoreet', 'move', 'pan'),
            ('Zoom', 'dolore magna aliquam', 'zoom_to_rect', 'zoom'),
            (None, None, None, None),
            ('Subplots', 'putamus parum claram', 'subplots', 'configure_subplots'),
            ('Save', 'sollemnes in futurum', 'filesave', 'save_figure'),
            ('Stack', 'sollemnes in futurum', 'back', 'stack_cameras'),
            )
        
        NavigationToolbar2Tk.__init__(self,canvas_,parent_)
        self.root = parent_
    
    def stack_cameras(self):
        self.root.update()


class MyApp(object):
    def __init__(self,root, data_paths):
        self.root = root
        self.data_paths = data_paths
        self._init_app()

    # here we embed the a figure in the Tk GUI
    def _init_app(self):
        self.fig, self.axs = plt.subplots(3,3,figsize=(8,8))
        """
        self.menu_item = tk.StringVar(self.root)
        self.menu_item.set('Option 1')
        self.menu = tk.OptionMenu(self.root, self.menu_item, *['Option 1', 'Option 2'], command=self.plot)
        self.menu.pack()
        """
        self.listbox = tk.Listbox(self.root, selectmode=tk.MULTIPLE)
        self.listbox.pack(side=tk.LEFT)
        options = ['cam 1','cam 2','cam 3','cam 4','cam 5','cam 6','cam 7','cam 8','cam 9','cam 10','cam 11','cam 12','cam 13','cam 14','cam 15','cam 16']
        for option in options:
            self.listbox.insert(tk.END, option)
        
        self.button = tk.Button(self.root, text="Replot", command=self.plot)
        self.button1 = tk.Button(self.root, text="Toggle Norm", command=self.toggle_norm)
        
        self.canvas = FigureCanvasTkAgg(self.fig,self.root)
        self.toolbar = EthogramWindow(self.canvas,self.root)
        self.toolbar.update()
        self.plot_widget = self.canvas.get_tk_widget()
        self.plot_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.toolbar.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
        self.canvas.draw()
        self.button.place(x=self.listbox.winfo_width()/2,y=self.listbox.winfo_height())
        self.button1.pack()
        
        self.internal = False
        
    # plot something random
    
    def toggle_norm(self):
        self.internal = not self.internal
        self.plot()
        
        
    def roll(self, x,  n):
        l = len(x)
        cpy = np.zeros(len(x))
        for i in range(0,l):
            sum = 0
            count = 0
            for j in range(-n,n+1):
                ind = i+j 
                if ind>=0 and ind<l:
                    sum+=x[ind]
                    count+=1
            sum/=count 
            cpy[i] = sum 
        return cpy



    def plot(self):
        
        self.stack = [5]
        selections = self.listbox.curselection()
        if len(selections)>0:
            self.stack = []
        for i in selections:
            self.stack.append(i+1)
            
        
        stack = self.stack
        vid_len = 18000
        fps = 10
        resolution = 3000
        
        lights_on = 410
        lights_off = 1130
        
        shift = True 
        shift_t = 0
        numdays = 1 

        manual_time = True
        
        if shift:
            shift_t = lights_on-360
            lights_on = 360
            lights_off = 1080
        
        vid_min_len = vid_len/fps/60
        
        fig = self.fig
        axs = self.axs
        
        
        directories = self.data_paths 
        records = []
        starts = []
        offset = 0
        offset1 = 0
        offset2 = 0
        coff = 0
        sk = 16
        for directory in directories:
            for root, dirs, files in os.walk(directory):
                for dir in dirs:
                    dirstring = dir.split('_') 
                    if len(dirstring)==3:
                        try:
                            cam = int(dirstring[1])
                        except:
                            continue
                        if not cam in stack:
                            continue
                        n = remove_leading_zeros(dirstring[2])
                        print(n+offset)
                        if n == 0:
                            starts.append(len(records))
                        if n == 0 and len(starts)==2:
                            offset = len(records)
                            offset1 = offset
                        elif n == 0 and len(starts)==3:
                            offset = len(records)
                            offset2 = offset
                        if offset2!=0:
                            if sk>0:
                                sk = sk-1
                                continue 
                            
                        records.append((dir, cam, n+offset, directory))
                
        behaviors = ['background','eating','drinking','rearing','climbing','digging','nesting','resting','grooming']
        
        behavior_colors = ['black','coral','goldenrod','blue','red','purple','mediumslateblue','darkslategray','green']
        
        
        LD = 6
        
        if True:
            for j in range(0,len(behaviors)):
                axs[int(j/3), int(j%3)].cla()
                axs[int(j/3), int(j%3)].set_xlim([0,48])
                axs[int(j/3), int(j%3)].set_xticks([0,6,12,18,24,30,36,42,48])
                axs[int(j/3), int(j%3)].set_ylim([0,1])
                axs[int(j/3), int(j%3)].yaxis.set_tick_params(labelleft=False)
                axs[int(j/3), int(j%3)].set_yticks([])
                axs[int(j/3), int(j%3)].axvspan(0,lights_on/1440*24,facecolor='darkgray',alpha=1)
                axs[int(j/3), int(j%3)].axvspan(lights_on/1440*24,lights_off/1440*24,facecolor='yellow',alpha=0.2)
                axs[int(j/3), int(j%3)].axvspan(lights_off/1440*24,1440/1440*24,facecolor='darkgray',alpha=1)
                axs[int(j/3), int(j%3)].axhspan(0,1,facecolor='white',alpha=1)
                axs[int(j/3), int(j%3)].axhspan(0,1,facecolor='gray',alpha=.3)
                axs[int(j/3), int(j%3)].set_title(behaviors[j], fontsize=10)
                if behaviors[j] == 'background':
                    axs[int(j/3), int(j%3)].set_title('motion', fontsize=10)
                
            
            days_drawn = False

            
                
            
            for s in stack:
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==s:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[2])
                print(cam_records)
                
                first_vid = os.path.join(os.path.join(cam_records[starts[0]][3], cam_records[starts[0]][0]), cam_records[starts[0]][0]+'.mp4')
                second_first_vid = os.path.join(os.path.join(cam_records[starts[0]][3], cam_records[starts[1]][0]), cam_records[starts[1]][0]+'.mp4')
                fct1 = 0
                fct2 = 0
                fct3 = 0
                if os.path.isfile(first_vid):
                    created_time1 = os.path.getctime(first_vid)
                    dt1 = datetime.fromtimestamp(created_time1).time()

                    created_time2 = os.path.getctime(second_first_vid)
                    dt2 = datetime.fromtimestamp(created_time2).time()
                    #fct1 = dt1.hour*60 + dt1.minute
                    #fct2 = dt2.hour*60 + dt2.minute
                    
                    if manual_time:
                        fct1 = 18*60+3
                        fct2 = 20*60+56
                        fct3 = 17*60+47
                
                
                behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                times = []
                subtimes = []
                offset_time = 0
                sub_behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                for i in range(0,len(cam_records)):
                    ts = (fct1+i*vid_min_len-shift_t)
                    predictions = os.path.join(os.path.join(cam_records[i][3], cam_records[i][0]), cam_records[i][0]+'_predictions.csv')
                    probs = os.path.join(os.path.join(cam_records[i][3], cam_records[i][0]), cam_records[i][0]+'_probs.csv')
                    print(predictions)
                    if os.path.isfile(predictions) and os.path.isfile(probs):
                        pred = pd.read_csv(predictions)
                        probs = pd.read_csv(probs)
                        sums = pred.sum(axis=0)
                        sums1 = np.mean(probs['motion'].values)
                        if i >= offset1:
                            ts = (ts-fct1+fct2) + 24*60
                        if i >= offset2 and offset2!=0:
                            ts = (ts-fct1+fct3)
                        times.append(ts)

                        for b in behaviors:
                            if b=='background':
                                height = sums1
                                behavior_counts[b].append(height)
                            else:
                                height = sums[b]
                                behavior_counts[b].append(height)

                        

                vid_min_len = 30

                        
                a = 0
                for b in behaviors:
                    bc = np.array(behavior_counts[b])
                    LD_b = LD

                    smoothed = bc
                    
                    if len(bc)==0:
                        continue
                    m = 0
                    try:
                        #m = np.median(smoothed[smoothed!=0])*2
                        m = smoothed.max()
                    except:
                        pass
                    self.internal = True
                    if self.internal:
                        if b == 'rearing':
                            m = smoothed[smoothed!=0].mean()*2
                        if m!=0:
                            smoothed = smoothed/m
                    else:
                        smoothed = smoothed/18000

                    if b == 'background':
                        smoothed = smoothed-smoothed.min()
                    
                    smoothed[smoothed>1] = 1

                    times = np.array(times)

                    mt = 0
                    try:
                        mt = times.max()/1440*24
                    except:
                        pass
                    numdays=int(mt/24)+1
                    step = 1/numdays
                    transparency = 0.7
                    if len(stack)>1:
                        transparency=1/len(stack)

                    smoothed_total = np.array([])
                    estr = False
                    if estr:
                        df_estr = pd.read_csv("C:\\Users\\Jones-Lab\\Desktop\\test_model\\Estrous_Exp1.csv")
                    for d in range(0,numdays):
                        valid = np.where(np.floor(times/1440)==d, True, False)
                        valid_next = np.where(np.floor(times/1440)==(d+1), True, False)
                        
                        
                        adj_times = np.copy(times)
                        adj_times[valid] = adj_times[valid]%1440


                        adj_times[valid_next] = adj_times[valid_next]%2880
                        
                        if d%2==1:
                            adj_times[valid_next] = adj_times[valid_next]+1440
                        
                        total = np.logical_or(valid, valid_next)
                        adj_times = adj_times[total]

                        
                        bot = np.ones(len(adj_times)) - step*(d+1)

                        if estr:
                            est = df_estr.iloc[d,s]
                            print(est)
                            est_alpha = 0
                            if est>1 and est<3:
                                est_alpha = .15
                            elif est>3:
                                est_alpha = .3
                            else:
                                est_alpha = 0
                            if est==-1:
                                axs[int(a/3), int(a%3)].axhspan(bot[0],bot[0]+1.0/numdays,0,1,color='white',alpha=est_alpha)
                            else:
                                axs[int(a/3), int(a%3)].axhspan(bot[0],bot[0]+1.0/numdays,0,1,color='black',alpha=est_alpha)

                        axs[int(a/3), int(a%3)].bar((adj_times)/1440*24,smoothed[total]/numdays,bottom=bot,width=vid_min_len/1440*24,color=behavior_colors[a],alpha=transparency)

                        
                        
                        
                        
                        smoothed_total= np.append(smoothed_total,smoothed)
                        
                        
                        
                        if not days_drawn:
                            if LD_b>0:
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0,xmax=.13, color='black',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.13,xmax=.38, color='yellow',alpha=.8)
                                LD_b-=1
                            else:
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0,xmax=.13, color='black',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.13,xmax=.38, color='gray',alpha=.8)

                            if LD_b>0:
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.38,xmax=.63, color='black',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.63,xmax=.88, color='yellow',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.88,xmax=1, color='black',alpha=.8)
                            else:
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.38,xmax=.63, color='black',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.63,xmax=.88, color='gray',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=(1-step*(d+1)),xmin=0.88,xmax=1, color='black',alpha=.8)



                    if not days_drawn:
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0,xmax=.13, color='black',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.13,xmax=.38, color='gray',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.38,xmax=.63, color='black',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.63,xmax=.88, color='gray',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.88,xmax=1, color='black',alpha=.8)
                    a+=1
                days_drawn = True

                
        title = "Cam"
        if len(stack)==1:
            title += " "+str(stack[0])
        else:
            title += "s "
            for e in range(0,len(stack)-1):
                title+=str(stack[e])+", "
            title+=str(stack[len(stack)-1])
        fig.suptitle(title, fontsize=16)
        fig.tight_layout(pad=1)
        self.fig.canvas.draw()
    


       
        


def remove_leading_zeros(num):
    for i in range(0,len(num)):
        if num[i]!='0':
            return int(num[i:])  
    return 0

  
            
if __name__ == '__main__':
    data_path1 = sys.argv[1]
    data_path2 = sys.argv[2]
    data_path3 = sys.argv[3]
    
    root = tk.Tk()
    app = MyApp(root, [data_path1, data_path2, data_path3])
    app.plot()
    root.mainloop()
