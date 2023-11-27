
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
from compare import circa_single
from CosinorPy import file_parser, cosinor, cosinor1
# from plot import circa_single_plot
# from generator import generate_data

import numpy as np
from scipy.optimize import least_squares

def circadian_model(t, amplitude, phase, period):
    t = np.array(t)
    return amplitude * np.cos(2 * np.pi * t / period + phase) + amplitude/2

def residuals(params, t, y):
    amplitude, phase, period = params
    return y - circadian_model(t, amplitude, phase, period)

def harmonic_regression(times, data):
    # Initial guesses
    amplitude_guess = (np.max(data) - np.min(data)) / 2
    phase_guess = 0
    period_guess = 24.0  # assuming a close to 24-hour cycle for circadian rhythms as an initial guess
    
    initial_params = [amplitude_guess, phase_guess, period_guess]
    
    # Fit the model
    result = least_squares(residuals, initial_params, args=(times, data), bounds=([-np.inf,-np.inf,20],[np.inf,np.inf,28]))
    
    amplitude, phase, period = result.x
    
    return amplitude, phase, period






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
    def __init__(self,root, data_path):
        self.root = root
        self.data_path = data_path
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

    def circ_rtest(self, alpha, w=None, d=0):
        """
        Computes Rayleigh test for non-uniformity of circular data.
        H0: the population is uniformly distributed around the circle
        HA: the population is not distributed uniformly around the circle
        
        Assumption: the distribution has maximally one mode and the data is
        sampled from a von Mises distribution!
        
        Input:
        alpha: sample of angles in radians
        w: number of incidences in case of binned angle data, default is None
        d: spacing of bin centers for binned data, if supplied
        correction factor is used to correct for bias in
        estimation of r, in radians. Default is 0
        
        Output:
        pval: p-value of Rayleigh's test
        z: value of the z-statistic
        """
        alpha = np.asarray(alpha).reshape(-1)
        #w = np.asarray(w).reshape(-1)

        print(len(alpha))
        print(len(w))
        
        if w is None:
            r = np.mean(np.exp(1j * alpha))
            n = len(alpha)
        else:
            if len(alpha) != len(w):
                raise ValueError("Input dimensions do not match.")
            w = np.asarray(w).reshape(-1)
            r = np.sum(w * np.exp(1j * alpha + d)) / np.sum(w)
            n = np.sum(w)

        # Compute Rayleigh's R
        R = n * np.abs(r)

        # Compute Rayleigh's z
        z = R**2 / n

        # Compute p value using approximation in Zar
        pval = np.exp(np.sqrt(1 + 4*n + 4*(n**2 - R**2)) - (1 + 2*n))

        return pval, z

    def plot(self):
        
        self.stack = [4]
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
        
        
        #fig_name = os.path.join(os.path.split(sys.argv[1])[0], 'ethogram.png')
        directory = self.data_path 
        records = []
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                dirstring = dir.split('_') 
                if len(dirstring)==3:
                    try:
                        cam = int(dirstring[1])
                    except:
                        continue
                    print(dirstring[2])
                    n = remove_leading_zeros(dirstring[2])
                    records.append((dir, cam, n))
                
        behaviors = ['background','eating','drinking','rearing','climbing','digging','nesting','resting','grooming']
        
        behavior_colors = ['black','coral','goldenrod','blue','red','purple','mediumslateblue','darkslategray','green']
        
        #behavior_groups = ['explore':['rearing','climbing','digging'], 'maintenance':['nesting', 'grooming'], 'sustenance':['eating', 'drinking'], 'rest':['resting']]
        
        LD = 1
        
        if False:
            c = 1
            while c<17:
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==c:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    c+=1
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[1])
                
                first_vid = os.path.join(os.path.join(directory, cam_records[0][0]), cam_records[0][0]+'.mp4')
                fct = 0
                if os.path.isfile(first_vid):
                    created_time = os.path.getctime(first_vid)
                    dt = datetime.fromtimestamp(created_time).time()
                    fct = dt.hour*60 + dt.minute

                    if manual_time:
                        created_time = 16*60+37
                
                for j in range(0,len(behaviors)):
                    axs[int(j/3), int(j%3)].cla()
                    axs[int(j/3), int(j%3)].set_xlim([0,1440/1440*24])
                    axs[int(j/3), int(j%3)].set_ylim([0,10])
                    axs[int(j/3), int(j%3)].axvspan(0,lights_on/1440*24,facecolor='darkgray',alpha=1)
                    axs[int(j/3), int(j%3)].axvspan(lights_on/1440*24,lights_off/1440*24,facecolor='yellow',alpha=0.2)
                    axs[int(j/3), int(j%3)].axvspan(lights_off/1440*24,1440/1440*24,facecolor='darkgray',alpha=1)
                    axs[int(j/3), int(j%3)].axhspan(0,9,facecolor='white',alpha=1)
                
                behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                times = []

                subtimes = []
                sub_behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                for i in range(0,len(cam_records)):
                    ts = (fct+i*vid_min_len)/1440*24
                    predictions = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_predictions.csv')
                    probs = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_probs.csv')
                    if os.path.isfile(predictions):
                        pred = pd.read_csv(predictions)
                        probs = pd.read_csv(probs)
                        sums = pred.sum(axis=0)
                        sums1 = probs.sum(axis=0)
                        times.append(ts)
                        for b in behaviors:
                            if b=='background':
                                height = sums1['motion']
                                behavior_counts[b].append(height)
                            else:
                                height = sums[b]
                                behavior_counts[b].append(height)
                        
                a = 0
                for b in behaviors:
                    bc = np.array(behavior_counts[b])
                    m = bc.max()
                    bc = bc/m * 9
                    axs[int(a/3), int(a%3)].bar(times,bc,width=vid_min_len,color=behavior_colors[a],alpha=0.7)
                    a+=1
                
                c+=1
        else:
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
                #axs[int(j/3), int(j%3)].axvline(x=24, color='black',alpha=.7, ls='--')
                
            behavior_averages = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
            behavior_average_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
            average_tr = 60
            zrs = np.zeros(int(1440/average_tr))
            
            days_drawn = False
            
            for b in behaviors:
                behavior_averages[b] = np.copy(zrs)
                behavior_average_counts[b] = np.copy(zrs)

            
                
            
            for s in stack:
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==s:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[2])
                
                first_vid = os.path.join(os.path.join(directory, cam_records[0][0]), cam_records[0][0]+'.mp4')
                fct = 0
                if os.path.isfile(first_vid):
                    created_time = os.path.getctime(first_vid)
                    dt = datetime.fromtimestamp(created_time).time()
                    fct = dt.hour*60 + dt.minute
                    
                    if manual_time:
                        fct = 12*60+9
                
                
                behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                times = []
                subtimes = []
                sub_behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                for i in range(0,len(cam_records)):
                    ts = (fct+i*vid_min_len-shift_t)
                    predictions = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_predictions.csv')
                    probs = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_probs.csv')
                    if os.path.isfile(predictions) and os.path.isfile(probs):
                        pred = pd.read_csv(predictions)
                        probs = pd.read_csv(probs)
                        sums = pred.sum(axis=0)
                        sums1 = np.mean(probs['motion'].values)
                        times.append(ts)

                        for b in behaviors:
                            if b=='background':
                                height = sums1
                                behavior_counts[b].append(height)
                            else:
                                height = sums[b]
                                behavior_counts[b].append(height)

                        ts_sub = 600
                        for tim in range(0,len(pred.iloc[:,0]),ts_sub):
                            end = tim + ts_sub 
                            if end>len(pred.iloc[:,0]):
                                end = len(pred.iloc[:,0])
                            subtimes.append(ts+tim/10/60)

                            s_0 = pred.iloc[tim:end,:].sum()
                            s_1 = probs.iloc[tim:end,:].sum()
                            for b in behaviors:
                                if b=='background':
                                    height = s_1['motion']
                                    sub_behavior_counts[b].append(height)
                                else:
                                    height = s_0[b]
                                    sub_behavior_counts[b].append(height)

                        # scale = 18000
                        # i = 0
                        # for st in range(0,len(pred["background"]),scale):
                        #     end = st+scale
                        #     if end>len(pred["background"]):
                        #         end = len(pred["background"])
                        #     times.append(ts+5*i)
                        #     for b in behaviors:
                        #         height = pred[b][st:end].sum()
                        #         behavior_counts[b].append(height)
                        #     i+=1
                vid_min_len = 30

                print(times)
                        
                a = 0
                stat_df = pd.DataFrame(columns=['period','amplitude','phase','angle of mean vec','pval of mean vec'], index=behaviors)
                for b in behaviors:
                    bc = np.array(behavior_counts[b])
                    LD_b = LD

                    start = 12.3833/24*2*np.pi
                    sub_bc = sub_behavior_counts[b]
                    time = np.linspace(start, len(sub_bc)/(1440)*2*np.pi+start, len(sub_bc))

                    df_clab = pd.DataFrame(columns=['angular time (pi=12hr)','counts'])
                    df_clab['angular time (pi=12hr)'] = time 
                    df_clab['counts'] = sub_bc

                    #df_clab.to_csv(os.path.join(os.getcwd(), "ClockLab/cam"+str(s)+"/"+b+".csv"))
                    ##########
                    calcstats = False
                    if calcstats:
                        sub_bc = sub_behavior_counts[b]
                        start = 12.3833/24*2*np.pi
                        #time = np.array(subtimes)/1440*2*np.pi
                        pvals = []
                        periods = []




                        # for tl in range(-360,360):
                        #     periods.append(tl)
                        time = np.linspace(start, len(sub_bc)/(1440)*2*np.pi+start, len(sub_bc))

                        # sumx = 0
                        # sumy = 0
                        
                        # for x in range(0,len(sub_bc)):
                        #     sumx += sub_bc[x]*np.cos(time[x])
                        #     sumy += sub_bc[x]*np.sin(time[x])
                              
                        # ang = math.atan2(sumy,sumx)
                        # magnitude = np.sqrt(sumy**2 + sumx**2)/len(sub_bc)
                        # R = magnitude
                        # print(R)
                        # # Calculate Rayleigh's Z
                        # Z = len(sub_bc) * R**2
                        # print(Z)
                        # # Calculate p-value
                        # pval = np.exp(-Z)

                        # stat_df['angle of mean vec'][b] = ((ang - 2*np.pi)/(2*np.pi)*24 - 24)%24
                        # stat_df['pval of mean vec'][b] = pval
                        # print("pval "+str(pval))

                        ## Normalized
                        sumx = 0
                        sumy = 0

                        # if np.median(sub_bc)!=0:
                        #     temp = sub_bc/(np.median(sub_bc))
                        # else:
                        #     temp = sub_bc
                        
                        for x in range(0,len(sub_bc)):
                            sumx += sub_bc[x]*np.cos(time[x])
                            sumy += sub_bc[x]*np.sin(time[x])
                              
                        ang = math.atan2(sumy,sumx)
                        #magnitude = np.sqrt(sumy**2 + sumx**2)/len(sub_bc)
                        #R = magnitude
                        # Calculate Rayleigh's Z
                        #Z = len(sub_bc) * R**2
                        # Calculate p-value
                        #pval = np.exp(-Z)
                        #pval = np.exp(np.sqrt(1+4*n+4*(n**2-R**2))-(1+2*n))

                        stat_df['angle of mean vec'][b] = ((ang - 2*np.pi)/(2*np.pi)*24 - 24)%24
                        print(time.shape)
                        print(np.array(sub_bc).shape)
                        pval, z = self.circ_rtest(time, np.array(sub_bc)/np.max(sub_bc))
                        stat_df['pval of mean vec'][b] = pval
                        print("pval "+str(pval))

                        fig, ax = plt.subplots(1, 1, constrained_layout=True)

                        residuals = []
                        ws = []
                        chars = []
                        from sympy import Matrix
                        for w in np.arange(.9,1.1,.01):


                            # Assuming t, w, and y are already defined as numpy arrays
                            n = len(time)

                            # Substitution
                            x = np.cos(w * time)
                            z = np.sin(w * time)
                            y = sub_bc

                            # Set up the normal equations
                            NE = np.array([
                                [n,       np.sum(x),    np.sum(z),    np.sum(y)],
                                [np.sum(x), np.sum(x**2), np.sum(x*z), np.sum(x*y)],
                                [np.sum(z), np.sum(x*z), np.sum(z**2), np.sum(z*y)]
                            ])

                            # Solve the normal equations using reduced row echelon form
                            # As numpy doesn't have a direct rref function, we'll use sympy's rref

                            RNE = Matrix(NE).rref()[0]
                            M, beta, gamma = RNE[:, 3]

                            Amp = np.sqrt(float(beta**2 + gamma**2))
                            theta = (math.atan2(gamma,beta)+2*np.pi)/(2*np.pi)*24-24

                            RSS = np.sum(np.power((y - (M + beta*x + gamma*z)),2))
                            residuals.append(RSS)
                            ws.append(w)
                            chars.append((Amp,theta))

                        residuals = np.array(residuals)
                        #print("Period "+str(ws[np.argmin(residuals)]*24))
                        per = ws[np.argmin(residuals)]*24
                        #print((ang-2*np.pi)/(2*np.pi)*24)
                        characteristics = chars[np.argmin(residuals)]
                        stat_df['period'][b] = per
                        stat_df['amplitude'][b] = characteristics[0]
                        stat_df['phase'][b] = characteristics[1]

                        # pvals = np.array(pvals)
                        # periods = np.array(periods)
                        #print(1/w[np.argmax(pgram)]*2*np.pi*24)

                        #plt.plot(periods, pvals)

                        # Sample time series data
                        #times = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
                        #values = np.array([2, 3, 5, 8, 10, 11, 10, 8, 6, 4, 2])

                        # Define period (For example, for a 24-hour cycle, the period is 24. Adjust as necessary.)
                        # If you want the period to be estimated, you might need to use more complex algorithms or libraries.
                        # fixed_period = 24

                        # # Perform cosinor analysis
                        # print(time)
                        # result = cosinor.periodogram(X=time, Y=sub_bc)

                        # # Extract parameters
                        # amplitude = result['amplitude']
                        # acrophase = result['acrophase']*12/np.pi  # phase in radians
                        # mesor = result['M']


                        # print("Amplitude:", amplitude)
                        # print("Acrophase:", acrophase)
                        # print("Mesor:", mesor)

                            

                    ##########
                    smoothed = bc
                    
                    if len(bc)==0:
                        continue
                    m = 0
                    try:
                        m = np.median(smoothed[smoothed!=0])*2
                        #print(b+" "+str(m))

                        m = smoothed.max()
                    except:
                        pass

                    #self.internal = True
                    if self.internal:
                        m = np.mean(smoothed[smoothed!=0])*2
                        print(b+" "+str(m))
                        if m!=0:
                            smoothed = smoothed/m
                    else:
                        smoothed = smoothed/18000

                    if b == 'background':
                        smoothed = smoothed-smoothed.min()
                    
                    smoothed[smoothed>.9] = .9
                    #dfcl = pd.DataFrame(bc)
                    #clocklab_file = os.path.join(os.getcwd(),os.path.join(os.path.join('ClockLab','cam'+str(s)), b+".csv"))
                    #dfcl.to_csv(clocklab_file)

                    times = np.array(times)

                    vids = np.arange(0,len(bc))
                    if b=='climbing':
                        bc_mean = np.mean(bc[bc>100])
                        # print(vids[bc>bc_mean])
                        # print(bc[vids[bc>bc_mean]])
                        # print(times[vids[bc>bc_mean]])
                        # print(times[vids[bc>bc_mean]]%1440/60)
                        # print(bc_mean)
                        
                    for t in range(0,len(times)):
                        ind = int((times[t]%1440)/average_tr)
                        behavior_averages[b][ind]+=bc[t]
                        
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

                    for d in range(0,numdays):
                        valid = np.where(np.floor(times/1440)==d, True, False)
                        valid_next = np.where(np.floor(times/1440)==(d+1), True, False)
                        
                        
                        adj_times = np.copy(times)
                        adj_times[valid] = adj_times[valid]%1440
                        
                        # do a quick circular stat analysis
                        heights = bc
                        start = adj_times[0]%1440
                        angles = np.copy(adj_times)/1440 * 2*math.pi
                        angles = angles - math.pi
                        
                        mH = heights.max()
                        real = np.where(heights>=0*mH, True, False)
                        
                        
                        sumx = 0
                        sumy = 0
                        
                        for t in range(0,len(heights)):
                            sumx += heights[t]*np.cos(angles[t])
                            sumy += heights[t]*np.sin(angles[t])
                            
                            
                        ang = math.atan2(sumy,sumx)
                        #print(ang)
                        if len(heights[real])!=0:
                            magnitude = np.sqrt(sumy**2 + sumx**2)/len(heights[real])
                        
                        
                        sigma = 200
                        if magnitude>0.0001:
                            sigma = 1/magnitude*20
                        adj_times[valid_next] = adj_times[valid_next]%2880
                        
                        if d%2==1:
                            adj_times[valid_next] = adj_times[valid_next]+1440
                        
                        total = np.logical_or(valid, valid_next)
                        adj_times = adj_times[total]

                        #print('day '+str(d)+' '+b+' '+str(np.mean(smoothed[valid])*m))
                        
                        bot = np.ones(len(adj_times)) - step*(d+1)
                        #m_day = bc[total].max()
                        #axs[int(a/3), int(a%3)].bar((adj_times)/1440*24,bc[total]/numdays/m_day,bottom=bot,width=vid_min_len/1440*24,color=behavior_colors[a],alpha=transparency)
                        
                        # #print(smoothed)

                        #smoothed = self.roll(bc[total],4)
                        #smooth_min = smoothed.min()
                        #smooth_max = smoothed.max()
                        #diff = smooth_max-smooth_min
                        #if diff!=0:
                        #    smoothed = (smoothed-smooth_min)*1/diff
                        axs[int(a/3), int(a%3)].bar((adj_times)/1440*24,smoothed[total]/numdays,bottom=bot,width=vid_min_len/1440*24,color=behavior_colors[a],alpha=transparency)
                        #axs[int(a/3), int(a%3)].plot((adj_times)/1440*24,smoothed[total]/numdays+bot,color=behavior_colors[a],alpha=transparency)
                        
                        
                        #print(ang+math.pi/(2*math.pi)*1440)
                        mean = (ang+math.pi)/(2*math.pi)*1440
                        if mean<720:
                            mean+=1440
                        
                        
                        smoothed_total= np.append(smoothed_total,smoothed)
                        
                        
                        #xvals = np.arange(0,2880,10)
                        #yvals = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-.5*((xvals-mean)/sigma)**2)*3
                        #my = yvals.max()
                        #if d==numdays-1:
                        #    axs[int(a/3), int(a%3)].plot(mean/2880*48, magnitude/.2*step,marker='x', color='black')
                        
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

                    # print(len(smoothed_total))
                    # if len(behavior_average_counts[b])==0:
                    #     behavior_average_counts[b] = smoothed_total
                    # else:
                    #     behavior_average_counts[b] += smoothed_total


                    if not days_drawn:
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0,xmax=.13, color='black',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.13,xmax=.38, color='gray',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.38,xmax=.63, color='black',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.63,xmax=.88, color='gray',alpha=.8)
                        axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.88,xmax=1, color='black',alpha=.8)
                    a+=1
                days_drawn = True

                stat_df.to_csv(os.path.join(os.getcwd(),"stats/cam"+str(s)+".csv"))
                
        #plt.tight_layout()
        title = "Cam"
        if len(stack)==1:
            title += " "+str(stack[0])
        else:
            title += "s "
            for e in range(0,len(stack)-1):
                title+=str(stack[e])+", "
            title+=str(stack[len(stack)-1])
        #fig.suptitle(title, fontsize=16)
        fig.tight_layout(pad=1)
        self.fig.canvas.draw()
        #self.root.after(600000, self.plot)
    


    def plot_individual_images(self):
        self.internal = False
        self.stack = [2]
        # selections = self.listbox.curselection()
        # if len(selections)>0:
        #     self.stack = []
        # for i in selections:
        #     self.stack.append(i+1)
            
        
        stack = self.stack
        vid_len = 18000
        fps = 10
        resolution = 3000
        
        lights_on = 410
        lights_off = 1130
        
        shift = True 
        shift_t = 0
        numdays = 1 
        
        if shift:
            shift_t = lights_on-360
            lights_on = 360
            lights_off = 1080
        
        vid_min_len = vid_len/fps/60
        
        fig = self.fig
        axs = self.axs
        
        
        #fig_name = os.path.join(os.path.split(sys.argv[1])[0], 'ethogram.png')
        directory = self.data_path 
        records = []
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                dirstring = dir.split('_') 
                if len(dirstring)>1:
                    cam = int(dirstring[1])
                    print(dirstring[2])
                    n = remove_leading_zeros(dirstring[2])
                    records.append((dir, cam, n))
                
        print(records)
        behaviors = ['background','eating','drinking','rearing','climbing','digging','nesting','resting','grooming']
        
        behavior_colors = ['black','red','green','sienna','goldenrod','purple','deeppink','blue','teal']
        
        #behavior_groups = ['explore':['rearing','climbing','digging'], 'maintenance':['nesting', 'grooming'], 'sustenance':['eating', 'drinking'], 'rest':['resting']]
        
        
        if len(stack)==0:
            c = 1
            while c<17:
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==c:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    c+=1
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[1])
                
                first_vid = os.path.join(os.path.join(directory, cam_records[0][0]), cam_records[0][0]+'.mp4')
                fct = 0
                if os.path.isfile(first_vid):
                    created_time = os.path.getctime(first_vid)
                    dt = datetime.fromtimestamp(created_time).time()
                    fct = dt.hour*60 + dt.minute
                
                for j in range(0,len(behaviors)):
                    axs[int(j/3), int(j%3)].cla()
                    axs[int(j/3), int(j%3)].set_xlim([0,1440/1440*24])
                    axs[int(j/3), int(j%3)].set_ylim([0,10])
                    axs[int(j/3), int(j%3)].axvspan(0,lights_on/1440*24,facecolor='darkgray',alpha=1)
                    axs[int(j/3), int(j%3)].axvspan(lights_on/1440*24,lights_off/1440*24,facecolor='yellow',alpha=0.2)
                    axs[int(j/3), int(j%3)].axvspan(lights_off/1440*24,1440/1440*24,facecolor='darkgray',alpha=1)
                    axs[int(j/3), int(j%3)].axhspan(0,9,facecolor='white',alpha=1)
                
                behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                times = []
                for i in range(0,len(cam_records)):
                    ts = (fct+i*vid_min_len)/1440*24
                    predictions = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_predictions.csv')
                    if os.path.isfile(predictions):
                        pred = pd.read_csv(predictions)
                        sums = pred.sum(axis=0)
                        times.append(ts)
                        for b in behaviors:
                            height = sums[b]
                            behavior_counts[b].append(height)
                        
                a = 0
                for b in behaviors:
                    bc = np.array(behavior_counts[b])
                    m = bc.max()
                    bc = bc/m * 9
                    axs[int(a/3), int(a%3)].bar(times,bc,width=vid_min_len,color=behavior_colors[a],alpha=0.7)
                    a+=1
                
                c+=1
        else:
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
                #axs[int(j/3), int(j%3)].axvline(x=24, color='black',alpha=.7, ls='--')
                
            behavior_averages = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
            behavior_average_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
            average_tr = 60
            zrs = np.zeros(int(1440/average_tr))
            
            days_drawn = False
            
            for b in behaviors:
                behavior_averages[b] = np.copy(zrs)
                #behavior_average_counts[b] = np.copy(zrs)
                
            
            for s in stack:
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==s:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[2])
                
                first_vid = os.path.join(os.path.join(directory, cam_records[0][0]), cam_records[0][0]+'.mp4')
                fct = 0
                if os.path.isfile(first_vid):
                    created_time = os.path.getctime(first_vid)
                    dt = datetime.fromtimestamp(created_time).time()
                    fct = dt.hour*60 + dt.minute
                
                for i in range(0, len(cam_records)):
                    
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
                    recs = cam_records[0:i+1]

                    behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                    times = []
                    for j in range(0,len(recs)):
                        ts = (fct+j*vid_min_len-shift_t)
                        predictions = os.path.join(os.path.join(directory, recs[j][0]), recs[j][0]+'_predictions.csv')
                        if os.path.isfile(predictions):
                            pred = pd.read_csv(predictions)
                            sums = pred.sum(axis=0)
                            times.append(ts)
                            for b in behaviors:
                                height = sums[b]
                                behavior_counts[b].append(height)
                            
                    a = 0
                    for b in behaviors:
                        bc = np.array(behavior_counts[b])
                        
                        if len(bc)==0:
                            continue
                        m = 0
                        try:
                            m = bc.max()
                        except:
                            pass
                        if self.internal:
                            #print(b + " : "+str(m))
                            if m!=0:
                                bc = bc/m
                        else:
                            bc = bc/vid_len
                            
                            
                        for t in range(0,len(times)):
                            ind = int((times[t]%1440)/average_tr)
                            behavior_averages[b][ind]+=bc[t]
                            #behavior_average_counts[b][ind]+=1
                            
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
                        for d in range(0,numdays):
                            valid = np.where(np.floor(times/1440)==d, True, False)
                            valid_next = np.where(np.floor(times/1440)==(d+1), True, False)
                            
                            
                            adj_times = np.copy(times)
                            adj_times[valid] = adj_times[valid]%1440
                            
                            # do a quick circular stat analysis
                            heights = bc
                            start = adj_times[0]%1440
                            angles = np.copy(adj_times)/1440 * 2*math.pi
                            angles = angles - math.pi
                            
                            mH = heights.max()
                            real = np.where(heights>=0*mH, True, False)
                            
                            
                            sumx = 0
                            sumy = 0
                            
                            for t in range(0,len(heights)):
                                sumx += heights[t]*np.cos(angles[t])
                                sumy += heights[t]*np.sin(angles[t])
                                
                                
                            ang = math.atan2(sumy,sumx)
                            #print(ang)
                            if len(heights[real])!=0:
                                magnitude = np.sqrt(sumy**2 + sumx**2)/len(heights[real])
                            
                            
                            sigma = 200
                            if magnitude>0.0001:
                                sigma = 1/magnitude*20
                            adj_times[valid_next] = adj_times[valid_next]%2880
                            
                            if d%2==1:
                                adj_times[valid_next] = adj_times[valid_next]+1440
                            
                            total = np.logical_or(valid, valid_next)
                            adj_times = adj_times[total]
                            
                            bot = np.ones(len(adj_times)) - step*(d+1)
                            axs[int(a/3), int(a%3)].bar((adj_times)/1440*24,bc[total]/numdays,bottom=bot,width=vid_min_len/1440*24,color=behavior_colors[a],alpha=transparency)
                            
                            #print(ang+math.pi/(2*math.pi)*1440)
                            mean = (ang+math.pi)/(2*math.pi)*1440
                            if mean<720:
                                mean+=1440
                            
                            
                            
                            
                            
                            #xvals = np.arange(0,2880,10)
                            #yvals = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-.5*((xvals-mean)/sigma)**2)*3
                            #my = yvals.max()
                            #if d==numdays-1:
                            #axs[int(a/3), int(a%3)].plot(mean/2880*48, magnitude/.2*step,marker='x', color='black', alpha=min((magnitude/.2),1))
                            
                            if not days_drawn:
                                axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0,xmax=.13, color='black',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.13,xmax=.38, color='yellow',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.38,xmax=.63, color='black',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.63,xmax=.88, color='yellow',alpha=.8)
                                axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.88,xmax=1, color='black',alpha=.8)
                        if not days_drawn:
                            axs[int(a/3), int(a%3)].axhline(y=0,xmin=0,xmax=.13, color='black',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.13,xmax=.38, color='yellow',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.38,xmax=.63, color='black',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.63,xmax=.88, color='yellow',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=0,xmin=0.88,xmax=1, color='black',alpha=.8)
                        a+=1
                    days_drawn = False
                    
                    #plt.tight_layout()
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
                    #self.fig.canvas.draw()
                    #plt.pause(.001)
                    self.fig.savefig(os.path.join(os.path.join(self.data_path, 'images'), "{:03d}.png".format(i)))



    def generate_individual_images(self):
        self.internal = True
        focus = 2
        
        
        #fig_name = os.path.join(os.path.split(sys.argv[1])[0], 'ethogram.png')
        directory = self.data_path 
        records = []
        count1 = 0
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                dirstring = dir.split('_') 
                if len(dirstring)>1:
                    cam = int(dirstring[1])
                    if cam!=focus:
                        continue
                    count = 0
                    video_file = os.path.join(self.data_path, os.path.join(dir, dir+".mp4"))
                    vidcap = cv2.VideoCapture(video_file)
                    success,image = vidcap.read()
                    while success:
                        if count%1000==0:
                            cv2.imwrite(os.path.join(self.data_path, os.path.join('frames', "%05d.png" % count1)), image)   
                            count1+=1  
                        success,image = vidcap.read()
                        count += 1
                    vidcap.release()

                
       
        


def remove_leading_zeros(num):
    for i in range(0,len(num)):
        if num[i]!='0':
            return int(num[i:])  
    return 0

  
            
if __name__ == '__main__':
    data_path = sys.argv[1]
    
    root = tk.Tk()
    app = MyApp(root, data_path)
    app.plot()
    root.mainloop()
"""
    vid_len = 18000
    fps = 10
    resolution = 3000
    
    lights_on = 410
    lights_off = 1130
    
    shift = True 
    shift_t = 0
    numdays = 1 
    
    if shift:
        shift_t = lights_on-360
        lights_on = 360
        lights_off = 1080
    
    vid_min_len = vid_len/fps/60
    
    fig, axs = plt.subplots(3,3,figsize=(17,10))
    
    while True:
    
        #fig_name = os.path.join(os.path.split(sys.argv[1])[0], 'ethogram.png')
        directory = data_path 
        records = []
        for root, dirs, files in os.walk(directory):
            for dir in dirs:
                dirstring = dir.split('_') 
                cam = int(dirstring[1])
                n = remove_leading_zeros(dirstring[2])
                records.append((dir, cam, n))
                
        behaviors = ['background','eating','drinking','rearing','climbing','digging','nesting','resting','grooming']
        behavior_colors = ['black','red','green','sienna','goldenrod','purple','deeppink','blue','teal']
        
        stack = [1,3,5,7]
        
        if len(stack)==0:
            c = 1
            while c<17:
                print(c)
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==c:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    c+=1
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[2])
                
                first_vid = os.path.join(os.path.join(directory, cam_records[0][0]), cam_records[0][0]+'.mp4')
                fct = 0
                if os.path.isfile(first_vid):
                    created_time = os.path.getctime(first_vid)
                    dt = datetime.fromtimestamp(created_time).time()
                    fct = dt.hour*60 + dt.minute
                
                for j in range(0,len(behaviors)):
                    axs[int(j/3), int(j%3)].cla()
                    axs[int(j/3), int(j%3)].set_xlim([0,1440/1440*24])
                    axs[int(j/3), int(j%3)].set_ylim([0,10])
                    axs[int(j/3), int(j%3)].axvspan(0,lights_on/1440*24,facecolor='darkgray',alpha=1)
                    axs[int(j/3), int(j%3)].axvspan(lights_on/1440*24,lights_off/1440*24,facecolor='yellow',alpha=0.2)
                    axs[int(j/3), int(j%3)].axvspan(lights_off/1440*24,1440/1440*24,facecolor='darkgray',alpha=1)
                    axs[int(j/3), int(j%3)].axhspan(0,9,facecolor='white',alpha=1)
                
                behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                times = []
                for i in range(0,len(cam_records)):
                    ts = (fct+i*vid_min_len)/1440*24
                    predictions = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_predictions.csv')
                    if os.path.isfile(predictions):
                        pred = pd.read_csv(predictions)
                        sums = pred.sum(axis=0)
                        times.append(ts)
                        for b in behaviors:
                            height = sums[b]
                            behavior_counts[b].append(height)
                        
                a = 0
                for b in behaviors:
                    bc = np.array(behavior_counts[b])
                    m = bc.max()
                    bc = bc/m * 9
                    axs[int(a/3), int(a%3)].bar(times,bc,width=vid_min_len,color=behavior_colors[a],alpha=0.7)
                    a+=1
                
                plt.pause(10)
                c+=1
        else:
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
                axs[int(j/3), int(j%3)].set_title(behaviors[j], fontsize=14, pad=2)
                axs[int(j/3), int(j%3)].axvline(x=24, color='black',alpha=.7, ls='--')
                
            behavior_averages = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
            behavior_average_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
            average_tr = 60
            zrs = np.zeros(int(1440/average_tr))
            
            days_drawn = False
            
            for b in behaviors:
                behavior_averages[b] = np.copy(zrs)
                behavior_average_counts[b] = np.copy(zrs)
                
            
            for s in stack:
                cam_records = []
                # Check each file
                for record in records:
                    if record[1]==s:
                        cam_records.append(record)
                
                if len(cam_records)==0:
                    continue
                cam_records = sorted(cam_records, key = lambda cr: cr[2])
                
                first_vid = os.path.join(os.path.join(directory, cam_records[0][0]), cam_records[0][0]+'.mp4')
                fct = 0
                if os.path.isfile(first_vid):
                    created_time = os.path.getctime(first_vid)
                    dt = datetime.fromtimestamp(created_time).time()
                    fct = dt.hour*60 + dt.minute
                
                
                
                behavior_counts = {'background':[],'eating':[],'drinking':[],'rearing':[],'climbing':[],'digging':[],'nesting':[],'resting':[],'grooming':[]}
                times = []
                for i in range(0,len(cam_records)):
                    ts = (fct+i*vid_min_len-shift_t)
                    predictions = os.path.join(os.path.join(directory, cam_records[i][0]), cam_records[i][0]+'_predictions.csv')
                    if os.path.isfile(predictions):
                        pred = pd.read_csv(predictions)
                        sums = pred.sum(axis=0)
                        times.append(ts)
                        for b in behaviors:
                            height = sums[b]
                            behavior_counts[b].append(height)
                        
                a = 0
                for b in behaviors:
                    bc = np.array(behavior_counts[b])
                    m = bc.max()
                    bc = bc/m
                    for t in range(0,len(times)):
                        ind = int((times[t]%1440)/average_tr)
                        behavior_averages[b][ind]+=bc[t]
                        behavior_average_counts[b][ind]+=1
                        
                    times = np.array(times)
                    mt = times.max()/1440*24
                    
                    numdays=int(mt/24)+1
                    step = 1/numdays
                    transparency = 0.7
                    if len(stack)>1:
                        transparency=1/len(stack)
                    for d in range(0,numdays):
                        valid = np.where(np.floor(times/1440)==d, True, False)
                        valid_next = np.where(np.floor(times/1440)==(d+1), True, False)
                        
                        adj_times = np.copy(times)
                        adj_times[valid] = adj_times[valid]%1440
                        adj_times[valid_next] = adj_times[valid_next]%2880
                        
                        total = np.logical_or(valid, valid_next)
                        adj_times = adj_times[total]
                        
                        bot = np.ones(len(adj_times)) - step*(d+1)
                        axs[int(a/3), int(a%3)].bar((adj_times)/1440*24,bc[total]/numdays,bottom=bot,width=vid_min_len/1440*24,color=behavior_colors[a],alpha=transparency)
                        
                        if not days_drawn:
                            axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0,xmax=.125, color='black',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.125,xmax=.375, color='yellow',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.375,xmax=.625, color='black',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.625,xmax=.875, color='yellow',alpha=.8)
                            axs[int(a/3), int(a%3)].axhline(y=step*(d+1),xmin=0.875,xmax=1, color='black',alpha=.8)
                            #axs[int(a/3), int(a%3)].axhline(y=step*(d+1), color='black',alpha=.3)
                    a+=1
                days_drawn = True
                
                
                plt.pause(.2) 
                
        plt.pause(300)
        print("replotting")
        
    
"""   
