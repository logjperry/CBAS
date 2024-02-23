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
import h5py
import cairo
from PIL import Image,ImageTk
from CosinorPy import file_parser, cosinor, cosinor1
from astropy.timeseries import LombScargle
import ruptures as rpt
from scipy.stats import t as tstat

import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib as mpl
from matplotlib import cm

import ttkbootstrap as ttk
import statsmodels.api as sm
import statsmodels.formula.api as smf

theme = 'superhero'

# class TransitionGraph:

#     def __init__(self, window, window_title, behaviors, path, color_list=None, width=800, height=800):
#         self.window = window
#         self.window.title(window_title)

#         self.behaviors = behaviors

#         self.width = width 
#         self.height = height

#         self.path = path

#         self.padding = 5
#         self.node_size = 50

#         self.bin_size = 1 
#         self.jump_size = 1

#         self.file = 'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\graph.png'

#         self.transitions(behaviors, path)

#         self.frame = tk.Frame(self.window)


#         self.color_list = [(255,97,3),(255,255,0),(65,105,225),(255,0,0),(255,20,147),(132,112,255),(59,59,59),(50,205,50),(138,54,15)]


#         self.canvas = tk.Canvas(self.frame, width=self.width, height=self.height)
#         self.canvas.grid(column=0, row=0)

#         self.frame.pack(anchor=tk.CENTER, pady=5, padx=5)
#         self.bin = tk.Scale(window, from_=1, to=20, orient=tk.HORIZONTAL)
#         self.bin.pack()
#         self.jump = tk.Scale(window, from_=1, to=120, orient=tk.HORIZONTAL)
#         self.jump.pack()
#         tk.Button(window, text='Show', command=self.update).pack(pady=5)


#         self.draw()
#         self.load_image()

        
#         self.window.mainloop()

#     def draw(self):
#         surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
#         ctx = cairo.Context(surface)

#         width = self.width
#         height = self.height

#         ctx.scale(width, height)

#         ctx.rectangle(0,0,1,1)
#         ctx.set_source_rgba(0, 0, 0, 1)
#         ctx.fill()
        
#         self.draw_graph(ctx, .3, self.behaviors, self.color_list, self.matrix)
#         self.draw_matrix(ctx, .75, self.behaviors, self.color_list, self.matrix)
        
#         surface.write_to_png(self.file)

#     def load_image(self):
#         img = ImageTk.PhotoImage(Image.open(self.file))
#         self.img = img
#         self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

#     def draw_graph(self, ctx, y, behaviors, colors, matrix):


#         node_size = .1

#         ex_padding = .05
#         in_padding = .02
#         node_size = (1-2*ex_padding - (len(behaviors)-1) * in_padding)/len(behaviors)

#         if node_size<0:
#             raise Exception('Padding too large for number of behaviors.')



#         cx = ex_padding + node_size/2
#         cy = y

#         roundness = .5
#         rc = (1-roundness)

#         Ialpha = .7
#         Oalpha = .5

#         ctx.set_line_cap(cairo.LINE_CAP_ROUND)

#         for i, b in enumerate(behaviors):

#             ctx.set_line_width(node_size*.1)
#             c = colors[i]

#             xl = cx-node_size/2
#             xr = cx+node_size/2
#             yt = cy-node_size/2
#             yb = cy+node_size/2

#             ctx.move_to(xl, cy+node_size/2*rc)
#             ctx.line_to(xl, cy-node_size/2*rc)
#             ctx.curve_to(xl, cy-node_size/2*rc, xl, yt, cx-node_size/2*rc, yt)
#             ctx.line_to(cx+node_size/2*rc, yt)
#             ctx.curve_to(cx+node_size/2*rc, yt, xr, yt, xr, cy-node_size/2*rc)
#             ctx.line_to(xr, cy+node_size/2*rc)
#             ctx.curve_to(xr, cy+node_size/2*rc, xr, yb, cx+node_size/2*rc, yb)
#             ctx.line_to(cx-node_size/2*rc, yb)
#             ctx.curve_to(cx-node_size/2*rc, yb, xl, yb, xl, cy+node_size/2*rc)


#             ctx.close_path()

#             ctx.set_source_rgba(c[0]/255, c[1]/255, c[2]/255, Ialpha)

#             ctx.fill_preserve()

#             ctx.set_source_rgba(0, 0, 0, Oalpha)
#             ctx.stroke()

#             ctx.select_font_face("Courier", cairo.FONT_SLANT_NORMAL)
#             ctx.set_font_size(node_size*.3)
            
#             (x, y, width, height, dx, dy) = ctx.text_extents(b)

#             ctx.move_to(cx - width/2, cy + height/4)    
#             ctx.set_source_rgba(0, 0, 0, 1)
#             ctx.show_text(b)


#             # draw the edges
#             tcx = cx 
#             tcy = cy 

#             edge_padding = node_size*.1

#             edge_step = (node_size-2*edge_padding)/(len(behaviors)-1)
            

#             for i1, b1 in enumerate(behaviors):

#                 dist = np.abs(i1-i)/len(behaviors)

#                 adjcx = (node_size + in_padding)*dist*len(behaviors)
#                 adjcx1 = -(node_size + in_padding)*dist*len(behaviors)

#                 stepup = .13


#                 # set the edge line weight to be smaller so that there is "padding"

#                 if b!=b1 and i1>i:

#                     weight = matrix[i,i1]

#                     if weight<.2:
#                         continue

#                     ctx.set_line_width(edge_step*weight)

#                     ctx.move_to(xl+i1*edge_step+edge_padding, yt-edge_padding)

#                     # left corner is (xl+i1*edge_step+edge_padding, yt-edge_padding-stepup*(dist))
#                     # right corner is ((xl+adjcx)+i*edge_step+edge_padding, yt-edge_padding-stepup*(dist))

#                     mx = (xl+i1*edge_step+edge_padding+(xl+adjcx)+i*edge_step+edge_padding)/2

#                     ctx.curve_to(xl+i1*edge_step+edge_padding, yt-edge_padding, xl+i1*edge_step+edge_padding, yt-edge_padding-stepup*(dist),mx, yt-edge_padding-stepup*(dist))
#                     ctx.curve_to(mx, yt-edge_padding-stepup*(dist), (xl+adjcx)+i*edge_step+edge_padding, yt-edge_padding-stepup*(dist),(xl+adjcx)+i*edge_step+edge_padding, yt-edge_padding)
                    
#                     ctx.set_source_rgba(c[0]/255, c[1]/255, c[2]/255, Ialpha)
#                     ctx.stroke()

#                     ctx.set_line_width(edge_step*2*weight)
#                     ctx.move_to((xl+adjcx)+i*edge_step+edge_padding, yt-edge_padding)
#                     ctx.line_to((xl+adjcx)+i*edge_step+edge_padding, yt-edge_padding)
#                     ctx.set_source_rgba(c[0]/255, c[1]/255, c[2]/255, Ialpha)
#                     ctx.stroke()

#                 elif b!=b1:
#                     weight = matrix[i,i1]

#                     if weight<.2:
#                         continue

#                     ctx.set_line_width(edge_step*weight)

#                     ctx.move_to(xl+i1*edge_step+edge_padding, yb+edge_padding)

#                     # left corner is (xl+i1*edge_step+edge_padding, yt-edge_padding-stepup*(dist))
#                     # right corner is ((xl+adjcx)+i*edge_step+edge_padding, yt-edge_padding-stepup*(dist))

#                     mx = (xl+i*edge_step+edge_padding+(xl+adjcx1)+i1*edge_step+edge_padding)/2

#                     ctx.curve_to(xl+i1*edge_step+edge_padding, yb+edge_padding, xl+i1*edge_step+edge_padding, yb+edge_padding+stepup*(dist),mx, yb+edge_padding+stepup*(dist))
#                     ctx.curve_to(mx, yb+edge_padding+stepup*(dist), (xl+adjcx1)+i*edge_step+edge_padding, yb+edge_padding+stepup*(dist),(xl+adjcx1)+i*edge_step+edge_padding, yb+edge_padding)


#                     ctx.set_source_rgba(c[0]/255, c[1]/255, c[2]/255, Ialpha)
#                     ctx.stroke()
                    
#                     ctx.set_line_width(edge_step*2*weight)
#                     ctx.move_to((xl+adjcx1)+i*edge_step+edge_padding, yb+edge_padding)
#                     ctx.line_to((xl+adjcx1)+i*edge_step+edge_padding, yb+edge_padding)
#                     ctx.set_source_rgba(c[0]/255, c[1]/255, c[2]/255, Ialpha)
#                     ctx.stroke()

#             # adjust the center x 
#             cx += node_size + in_padding
    
#     def draw_matrix(self, ctx, y, behaviors, colors, matrix):

#         map = sns.color_palette("viridis", as_cmap=True).colors
#         map_len = len(map)

#         width = self.width
#         height = self.height

#         node_size = .04

#         padding = 0.003 

#         size = node_size*len(behaviors)+padding*(len(behaviors)-1)

#         cx = .5+node_size/2+padding

#         xl = cx - size/2
#         xr = cx + size/2

#         yt = y - size/2
#         yb = y + size/2

#         border_size = padding
#         ctx.set_line_width(border_size)

#         # ctx.rectangle(xl-padding,yt-padding,size+padding*2,size+padding*2)
#         # ctx.set_source_rgba(50/255, 50/255, 50/255, 1)
#         # ctx.fill()
#         # ctx.stroke()

#         for c in range(len(behaviors)):
#             cl = colors[c]
#             txl = xl-padding + c*(node_size+padding)
#             tyl = yt-padding*2.5-node_size

#             # ctx.rectangle(txl,tyl,node_size+padding*2,node_size+padding*3)
#             # ctx.set_source_rgba(50/255, 50/255, 50/255, 1)
#             # ctx.fill()
#             # ctx.stroke()

            
#             # ctx.rectangle(txl+padding,tyl+padding,node_size,node_size)
#             # ctx.set_source_rgba(1, 1, 1, 1)
#             # ctx.fill()
#             # ctx.stroke()

            
#             ctx.rectangle(txl+padding,tyl+padding,node_size,node_size)
#             ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, .7)
#             ctx.fill_preserve()
#             ctx.set_source_rgba(0, 0, 0, .5)
#             ctx.stroke()

#         for c in range(len(behaviors)):
#             cl = colors[c]
#             txl = xl-padding -padding*1.5-node_size
#             tyl = yt-padding + c*(node_size+padding)

#             # ctx.rectangle(txl,tyl,node_size+padding*3,node_size+padding*2)
#             # ctx.set_source_rgba(50/255, 50/255, 50/255, 1)
#             # ctx.fill()
#             # ctx.stroke()

            
#             # ctx.rectangle(txl+padding,tyl+padding,node_size,node_size)
#             # ctx.set_source_rgba(1, 1, 1, 1)
#             # ctx.fill()
#             # ctx.stroke()

            
#             ctx.rectangle(txl+padding,tyl+padding,node_size,node_size)
#             ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, .7)
#             ctx.fill_preserve()
#             ctx.set_source_rgba(0, 0, 0, .5)
#             ctx.stroke()

#         i = 0
#         for x in np.arange(0, size, node_size+padding):
#             j = 0
#             for y in np.arange(0, size, node_size+padding):
#                 weight = matrix[j,i]
#                 c = colors[j]
#                 # ctx.rectangle(x+xl,y+yt,node_size,node_size)
#                 # ctx.set_source_rgba(0, 0, 0, .5)
#                 # ctx.fill()
#                 # ctx.stroke()

#                 ctx.rectangle(x+xl,y+yt,node_size,node_size)
#                 ctx.set_source_rgba(map[int(weight*map_len)][0],map[int(weight*map_len)][1],map[int(weight*map_len)][2], .7)
#                 ctx.fill()
#                 ctx.stroke()
#                 j+=1
#             i+=1

#     def transitions(self, behaviors, path):

#         df = pd.read_csv(path)
#         df = df.to_numpy()[1:,1:]

#         matrix = np.zeros((len(behaviors), len(behaviors)))

#         linear = []

#         for i in range(len(df)):
#             linear.append(np.argmax(df[i]))
        
#         # calc the transitions
#         linear = np.array(linear)
#         bin_size = self.bin_size

#         jump_size = self.jump_size

#         binned = []

#         for i in range(0,len(linear),bin_size):

#             end = i+bin_size
#             if end>len(linear):
#                 end = len(linear)

#             chunk = linear[i:end]
#             counts = []
#             for b in range(len(behaviors)):
#                 counts.append(np.sum(chunk==b))
#             if counts[np.argmax(counts)]==0:
#                 binned.append(-1)
#             else:
#                 binned.append(np.argmax(counts))
        
#         instances = [0 for b in behaviors]
#         for i in range(len(binned)):
#             instances[binned[i]] += 1

#         for b in range(len(behaviors)):
#             next = [0 for b1 in behaviors]
#             for i in range(len(binned)-jump_size):
#                 if b==binned[i]:
#                     next[binned[i+jump_size]] += 1
#             next = np.array(next)
#             next = next/instances[b]

#             matrix[b,:] = next
        
#         self.matrix = matrix

#     def update(self):

#         self.bin_size = int(self.bin.get())
#         self.jump_size = int(self.jump.get())

#         self.transitions(self.behaviors, self.path)
#         self.draw()
#         self.load_image()

class TransitionRaster:

    def __init__(self, window, window_title, behaviors, paths, starts, name, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)], width=800, height=800, main=True):
        self.window = window

        self.behaviors = behaviors

        self.width = width 
        self.height = height

        self.paths = paths

        self.padding = 5
        self.node_size = 50

        self.bin_size = 10 
        self.jump_size = 1
        self.power = 1

        self.main = main

        self.starts = starts

        self.exps = {os.path.split(paths[p])[1]:[starts[p], paths[p]] for p in range(len(paths))}
       

        self.file = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\raster_{name}.png'  
        self.file1 = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\internal_{name}.csv'  
        self.group_names = []

        self.pop_values = {b:np.zeros((len(behaviors), len(behaviors))).flatten() for b in np.arange(0, 24, .5)}

        self.transitions(behaviors, paths)


        self.group_name = self.group_names

        self.frame = tk.Frame(self.window)

        self.color_list = color_list


        self.canvas = tk.Canvas(self.frame, width=self.width, height=self.height)
        self.canvas.grid(column=0, row=0)

        self.frame.pack(anchor=tk.CENTER, pady=5, padx=5)
        self.bin = tk.Scale(window, from_=1, to=600, orient=tk.HORIZONTAL)
        self.bin.pack()
        self.jump = tk.Scale(window, from_=1, to=120, orient=tk.HORIZONTAL)
        self.jump.pack()
        tk.Button(window, text='Show', command=self.change).pack(pady=5)
        tk.Button(window, text='Internal Var', command=self.calc_internal_var).pack(pady=5)

        
        options = self.group_names
        self.listbox = tk.Listbox(self.window, selectmode = "multiple")
        self.listbox.pack(side=tk.LEFT)
        
        for option in options:
            self.listbox.insert(tk.END, option)
        
        self.button = tk.Button(self.window, text="Replot", command=self.update).pack(side=tk.BOTTOM)



        self.draw()
        self.load_image()

        if main:
            self.window.mainloop()

    def draw(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)
        
        self.draw_raster(ctx, .5, self.behaviors, self.color_list)
        
        surface.write_to_png(self.file)

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
        image = Image.open(self.file)
        if not self.main:
            image = self.resize_image(image, 400, 400)
        img = ImageTk.PhotoImage(image)
        self.img = img
        self.canvas.config(width=image.width, height=image.height)
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    def draw_raster(self, ctx, y, behaviors, colors, LD=True):

        opacity = 2/len(self.group_name)
        for group in range(len(self.group_name)):

            first = False 
            last = False 
            if group == len(self.group_name)-1:
                last = True

            gn = self.group_name[group]

            if group==0:
                first=True 
                self.pop_values = {b:np.zeros((len(behaviors), len(behaviors))).flatten() for b in np.arange(0, 24, .5)}

            matrix = np.array(self.transition_data[gn][1])
            times = np.array(self.transition_data[gn][0])
            


            width = self.width
            height = self.height

            vraster_size = .08
            hraster_size = .8

            padding = 0.01 

            vsize = vraster_size*len(behaviors)+padding*(len(behaviors)-1)
            hsize = hraster_size

            cx = .5

            xl = cx - hsize/2
            xr = cx + hsize/2

            yt = y - vsize/2
            yb = y + vsize/2


            ctx.set_line_width(.005)
            s_padding = padding*.2

            bn_size = hraster_size*.1
            sn_width = ((hsize - bn_size) - padding)/len(self.pop_values)
            sn_height = ((vraster_size) - s_padding*(len(behaviors)-1))/(len(behaviors)-1)
            txr = xl + bn_size + padding

            if first:
                
                for c in range(len(behaviors)):
                    tyl = yt + c*(vraster_size+padding)

                    for c1 in range(len(behaviors)):

                        if c!=c1:

                            nc1 = c1
                            if c1>c:
                                nc1 = c1-1

                            ttxl = txr-padding/2
                            ttyl = tyl+nc1*(sn_height+s_padding)
                                    
                            cl = colors[c1]
                            ctx.rectangle(ttxl,ttyl,padding,sn_height)
                            ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, 1)
                            ctx.fill()


            # lay the background time
            if first:
                if LD:
                    ctx.rectangle(txr+padding+(12/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
                    ctx.set_source_rgb(223/255,223/255,223/255)
                    ctx.fill()
                    
                    ctx.set_source_rgb(255/255, 239/255, 191/255)
                    ctx.rectangle(txr+padding+(0/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
                    ctx.fill()
                else:
                    ctx.rectangle(txr+padding+(12/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
                    ctx.set_source_rgb(223/255,223/255,223/255)
                    ctx.fill()
                    
                    ctx.set_source_rgb(246/255, 246/255, 246/255)
                    ctx.rectangle(txr+padding+(0/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
                    ctx.fill()
                
                for c in range(len(behaviors)):
                    cl = colors[c]
                    txl = xl 
                    tyl = yt + c*(vraster_size+padding)

                    
                    ctx.rectangle(txl,tyl,bn_size,vraster_size)
                    ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, 1)
                    ctx.fill()
                    if c!=len(behaviors)-1:
                        ctx.set_line_width(0.002)
                        ctx.move_to(txr-padding/2,tyl+vraster_size+padding/2)
                        ctx.line_to(txr+(24/24*len(matrix))*sn_width+padding,tyl+vraster_size+padding/2)
                        ctx.set_source_rgba(0, 0, 0, 1)
                        ctx.stroke()


            
            
            for t in range(len(matrix)):
                weights = np.array(matrix[t])

                adj_time = times[t]

                bin_time = float(np.floor(adj_time*2)/2)

                self.pop_values[bin_time] += weights/len(self.group_name)
            
            if last:
                times = list(self.pop_values.keys())
                times.sort()

                mws = []

                for c in range(len(behaviors)):
                    tyl = yt + c*(vraster_size+padding)

                    msw = 0
                    for c1 in range(len(behaviors)):

                        if c!=c1:

                            nc1 = c1
                            if c1>c:
                                nc1 = c1-1
                            for t in range(len(times)):
                                adj_time = times[t]
                                w = self.pop_values[adj_time][c*len(behaviors)+c1]

                                if w>msw:
                                    msw = w 
                    

                    if msw==0:
                        msw = 1

                    mws.append(msw)


                for c in range(len(behaviors)):
                    tyl = yt + c*(vraster_size+padding)

                    for c1 in range(len(behaviors)):

                        if c!=c1:

                            nc1 = c1
                            if c1>c:
                                nc1 = c1-1



                            for t in range(len(times)):
                                # # transition c to c1 at time t
                                # # calculate the average transition from x to c1, excluding c and c1
                                # transit = []
                                # for x in range(len(behaviors)):
                                #     if x==c or x==c1:
                                #         continue 
                                #     transit.append(matrix[t,x*len(behaviors)+c1])
                                # transit = np.array(transit)
                                # avg_transition = np.mean(transit)
                                # std_transition = np.std(transit)
                                    
                                cl = colors[c1]

                                adj_time = times[t]
                                weight = self.pop_values[adj_time][c*len(behaviors)+c1]/mws[c]

                                ttxl = txr+padding+(adj_time/24*len(matrix))*sn_width

                                ttyl = tyl+nc1*(sn_height+s_padding)


                                ctx.rectangle(ttxl,ttyl,sn_width,sn_height)
                                ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, (weight)*1)
                                ctx.fill()
            
    def calc_internal_var(self):

        original_gn = self.group_name
        self.draw()

        running = []

        for i in range(len(original_gn)):

            a = [original_gn[i]]
            b = [original_gn[j] for j in range(len(original_gn)) if j!=i]

            self.group_name = a 
            self.draw()
            m1 = self.pop_values.copy()
            self.group_name = b 
            self.draw()
            m2 = self.pop_values.copy()
            

            times = list(self.pop_values.keys())
            times.sort()
            t1 = []
            t2 = []
            for t in times:
                t1.append(m1[t])
                t2.append(m2[t])
            t1 = np.array(t1)
            t2 = np.array(t2)

            matrix = t2-t1
            diff = []
            for m in range(len(matrix)):
                diff.append([np.mean(np.absolute(matrix[m]))])

            columns = times
            difs = []
            for t in range(len(times)):
                beh = []
                for c in range(len(self.behaviors)):

                    ws = []
                    for c1 in range(len(self.behaviors)):
                        weight = (matrix[t,c*len(self.behaviors)+c1])
                        ws.append(np.absolute(weight))
                    beh.append(np.mean(np.array(ws)))
                difs.append(beh)

            diff = np.array(diff)
            difs = np.array(difs)

            total = np.concatenate((diff, difs), axis=1)
            total = np.transpose(total).tolist()

            running.extend(total)

        full_indices = []
        indices = ['total']
        indices.extend(self.behaviors)
        for g in original_gn:
            full_indices.extend([i+' '+g for i in indices])


        df = pd.DataFrame(data=running, columns=columns, index=full_indices)

        df.to_csv(self.file1)

    def transitions(self, behaviors, paths):

        video_size = 18000
        videosperhour = 36000/video_size

        videosintf = 24*videosperhour

        groups = {}
        group_names = []
        for path in paths:


            exp = os.path.split(path)[1]

            f = []

            for (dirpath, dirnames, filenames) in os.walk(path):
                f.extend(dirnames)
                break

            

            if os.path.exists(os.path.join(path, f[0], f[0]+'_outputs_inferences.csv')):
                files = {fold:os.path.join(path, fold, fold+'_outputs_inferences.csv') for fold in f}
            elif os.path.exists(os.path.join(path, f[0], f[0]+'_predictions.csv')):
                files = {fold:os.path.join(path, fold, fold+'_predictions.csv') for fold in f}
            else:
                raise Exception('No valid files found in '+path)



            for file in files.keys():
                try:
                    name = file.split('_')[1]
                except:
                    continue

                if exp+' - '+name in group_names:
                    groups[exp+' - '+name].append(files[file])
                else:
                    group_names.append(exp+' - '+name)
                    groups[exp+' - '+name] = [files[file]]

        transition_data = {}
    
        self.group_names = group_names

        for g in group_names:
            
            files = groups[g]

            expname = g.split(' - ')[0]
            start = self.exps[expname][0]

            time_data = {}
            m = 0
            for file in files:

                time_data[remove_leading_zeros(os.path.split(file)[1].split('_')[2])] = []

                matrix = np.zeros((len(behaviors), len(behaviors)))
                df = pd.read_csv(file)
                df = df.to_numpy()[1:,1:]

                linear = []

                for i in range(len(df)):
                    linear.append(np.argmax(df[i]))
                
                # calc the transitions
                linear = np.array(linear)
                bin_size = self.bin_size

                jump_size = self.jump_size

                binned = []

                for i in range(0,len(linear),bin_size):

                    end = i+bin_size
                    if end>len(linear):
                        end = len(linear)

                    chunk = linear[i:end]
                    counts = []
                    for b in range(len(behaviors)):
                        counts.append(np.sum(chunk==b))
                    binned.append(counts)
                
                instances = [0 for b in behaviors]
                for i in range(len(binned)):
                    for b in range(len(behaviors)):
                        instances[b] += binned[i][b]

                for b in range(len(behaviors)):
                    next = np.zeros(len(behaviors))
                    total_insts = 0
                    for i in range(len(binned)-jump_size):
                        insts = binned[i][b]
                        n = np.array(binned[i+jump_size])

                        factor = insts*n/bin_size

                        next += factor
                        total_insts += insts
                        
                    if total_insts!=0:
                        next = next/total_insts
                    else:
                        next[b] = 1

                    matrix[b,:] = next
                
                time_data[remove_leading_zeros(os.path.split(file)[1].split('_')[2])] = matrix.flatten()

                if remove_leading_zeros(os.path.split(file)[1].split('_')[2])>m:
                    m = remove_leading_zeros(os.path.split(file)[1].split('_')[2])
        
            # calculate times
                    
            values = []
            for i in range(m):
                try:
                    value = time_data[i]
                    if len(value)==0:
                        raise Exception
                    values.append(value)
                except:
                    value = np.zeros((len(behaviors), len(behaviors))).flatten()
                    values.append(value)

            fold_days = 1

            folded = {t:np.zeros((len(behaviors), len(behaviors))).flatten() for t in np.arange(0, fold_days*24*videosperhour, 1)}
            count = {t:0 for t in np.arange(0, fold_days*24*videosperhour, 1)}

            for t in range(len(values)):
                val = values[t]
                folded[t%(24*fold_days*videosperhour)] += val
                count[t%(24*fold_days*videosperhour)] += 1
            
            dt = 1/videosperhour
            times = []
            values = []
            
            keys = list(folded.keys())
            keys.sort()

            for k in keys:
                if count[k]==0:
                    count[k] = 1

                folded[k] = folded[k]/count[k]
                times.append(k*dt)
                values.append(folded[k])


            times = np.array(times)
            times = (times + start - 6) % (fold_days*24)
            
            transition_data[g] = (times, values)
        
        self.transition_data = transition_data
    
    def update(self):
        
        selections = self.listbox.curselection()
        
        self.group_name = [self.group_names[g] for g in selections]
        
        self.draw()
        self.load_image()

    def updatePops(self):
        self.draw()
        self.load_image()

    def change(self):

        self.bin_size = int(self.bin.get())
        self.jump_size = int(self.jump.get())

        self.transitions(self.behaviors, self.paths)
        self.draw()
        self.load_image()

class TransitionRasterDif:

    def __init__(self, window, window_title, behaviors, paths1, paths2, starts1, starts2, color1, color2, name1, name2, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)], width=800, height=800):
        self.window = window

        self.behaviors = behaviors

        self.width = width 
        self.height = height

        self.paths1 = paths1
        self.paths2 = paths2

        self.padding = 5
        self.node_size = 50

        self.bin_size = 300 
        self.jump_size = 10
        self.power = 10

        self.starts1 = starts1
        self.starts2 = starts2

        self.color1 = color1 
        self.color2 = color2

        self.exp1 = os.path.split(paths1[0])[1]
        self.exp2 = os.path.split(paths2[0])[1]


        self.file = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\raster_{name1}_{name2}.png'
        self.file1 = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\diff_{name1}_{name2}.csv'
        self.file2 = f'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\diff_{name2}_{name1}.csv'

        self.frame1 = tk.Frame(self.window)
        self.frame2 = tk.Frame(self.window)
        self.frame3 = tk.Frame(self.window)

        self.color_list = color_list
        self.r1 = TransitionRaster(self.frame1, 'Raster 1', behaviors, paths1, starts1, name1, color_list, width=800, height=800, main=False)
        self.r2 = TransitionRaster(self.frame2, 'Raster 2', behaviors, paths2, starts2, name2, color_list, width=800, height=800, main=False)

        
        self.canvas = tk.Canvas(self.frame3, width=self.width, height=self.height)
        self.canvas.pack()

        self.draw()
        self.load_image()

        self.frame1.pack(side='left', pady=5, padx=5)
        self.frame2.pack(side='right', pady=5, padx=5)
        self.frame3.pack(side='right', pady=5, padx=5)




        self.button = tk.Button(self.frame3, text="Replot Difference", command=self.update).pack(side=tk.BOTTOM, anchor='center')
        self.button1 = tk.Button(self.frame3, text="Difference CSV", command=self.generate_differences).pack(side=tk.BOTTOM, anchor='center')


        
        self.window.mainloop()

    def draw(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)
        
        self.draw_raster(ctx, .5, self.behaviors, self.color_list)
        
        surface.write_to_png(self.file)

    def load_image(self):
        img = ImageTk.PhotoImage(Image.open(self.file))
        self.img = img
        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    def draw_raster(self, ctx, y, behaviors, colors, LD=True):

        map = sns.diverging_palette(self.color2, self.color1, s=100, l=50, as_cmap=True)
        map_len = 256
        self.cmap = plt.get_cmap(map)
        self.norm = mpl.colors.Normalize(vmin=0, vmax=256)

        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

        def get_rgb(val):
            return self.scalarMap.to_rgba(val)
        

        map1 = sns.color_palette("viridis", as_cmap=True).colors
        map_len1 = len(map1)

        # get the binned averages from each raster
        pop1 = self.r1.pop_values
        pop2 = self.r2.pop_values

        times = list(pop1.keys())
        times.sort()


        matrix1 = np.array([pop1[t] for t in times])
        matrix2 = np.array([pop2[t] for t in times])

        matrix = matrix1 - matrix2

        width = self.width
        height = self.height

        vraster_size = .08
        hraster_size = .8

        padding = 0.01 

        vsize = vraster_size*len(behaviors)+padding*(len(behaviors)-1)
        hsize = hraster_size

        cx = .5

        xl = cx - hsize/2
        xr = cx + hsize/2

        yt = y - vsize/2
        yb = y + vsize/2


        ctx.set_line_width(.005)
        s_padding = padding*.2

        bn_size = hraster_size*.1
        sn_width = ((hsize - bn_size) - padding)/len(times)
        sn_height = ((vraster_size) - s_padding*(len(behaviors)-1))/(len(behaviors)-1)
        txr = xl + bn_size + padding

        

        for c in range(len(behaviors)):
            tyl = yt + c*(vraster_size+padding)

            for c1 in range(len(behaviors)):

                if c!=c1:

                    nc1 = c1
                    if c1>c:
                        nc1 = c1-1

                    ttxl = txr-padding/2
                    ttyl = tyl+nc1*(sn_height+s_padding)
                            
                    cl = colors[c1]
                    ctx.rectangle(ttxl,ttyl,padding,sn_height)
                    ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, 1)
                    ctx.fill()

                    


        # lay the background time
        if LD:
            ctx.rectangle(txr+padding+(12/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
            ctx.set_source_rgb(223/255,223/255,223/255)
            ctx.fill()
            
            ctx.set_source_rgb(255/255, 239/255, 191/255)
            ctx.rectangle(txr+padding+(0/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
            ctx.fill()
        else:
            ctx.rectangle(txr+padding+(12/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
            ctx.set_source_rgb(223/255,223/255,223/255)
            ctx.fill()
            
            ctx.set_source_rgb(246/255, 246/255, 246/255)
            ctx.rectangle(txr+padding+(0/24*len(times))*sn_width,yt-4*(sn_height+s_padding),(12/24*len(times))*sn_width,4*(sn_height+s_padding))
            ctx.fill()

        
        
        for c in range(len(behaviors)):
            cl = colors[c]
            txl = xl 
            tyl = yt + c*(vraster_size+padding)

            
            ctx.rectangle(txl,tyl,bn_size,vraster_size)
            ctx.set_source_rgba(cl[0]/255, cl[1]/255, cl[2]/255, 1)
            ctx.fill()
            if c!=len(behaviors)-1:
                ctx.set_line_width(0.002)
                ctx.move_to(txr-padding/2,tyl+vraster_size+padding/2)
                ctx.line_to(txr+(24/24*len(matrix))*sn_width+padding,tyl+vraster_size+padding/2)
                ctx.set_source_rgba(0, 0, 0, 1)
                ctx.stroke()

        mws = []

        for c in range(len(behaviors)):
            tyl = yt + c*(vraster_size+padding)

            msw = 0
            for c1 in range(len(behaviors)):

                if c!=c1:

                    nc1 = c1
                    if c1>c:
                        nc1 = c1-1
                    for t in range(len(times)):
                        w = matrix[t,c*len(behaviors)+c1]

                        if w>msw:
                            msw = w 
            

            if msw==0:
                msw = 1

            mws.append(msw)

        mw = np.max(np.array(mws))
        
        for c in range(len(behaviors)):
            tyl = yt + c*(vraster_size+padding)

            for c1 in range(len(behaviors)):

                if c!=c1:

                    nc1 = c1
                    if c1>c:
                        nc1 = c1-1

                    for t in range(len(times)):
                        # # transition c to c1 at time t
                        # # calculate the average transition from x to c1, excluding c and c1
                        # transit = []
                        # for x in range(len(behaviors)):
                        #     if x==c or x==c1:
                        #         continue 
                        #     transit.append(matrix[t,x*len(behaviors)+c1])
                        # transit = np.array(transit)
                        # avg_transition = np.mean(transit)
                        # std_transition = np.std(transit)
                            
                        cl = colors[c1]
                        weight = (matrix[t,c*len(behaviors)+c1])/mw

                        adj_time = times[t]

                        ttxl = txr+padding+(adj_time/24*len(matrix))*sn_width
                        ttyl = tyl+nc1*(sn_height+s_padding)


                        ctx.rectangle(ttxl,ttyl,sn_width,sn_height)
                        index = int(map_len*((weight+1)/2))
                        if index==map_len:
                            index = index-1
                            
                        ctx.set_source_rgba(get_rgb(index)[0], get_rgb(index)[1], get_rgb(index)[2], 1)
                        ctx.fill()
        for y in np.arange(0,vsize,sn_height):
            ttxl = txr+padding+(24/24*len(matrix))*sn_width + 1.75*padding
            f = 1.5

            if y == (vsize-sn_height):
                f = 1

            ctx.rectangle(ttxl,y+yt,sn_width*2,sn_height*f)

            index = int(y/vsize*map_len)
            if index==map_len:
                index = index-1
                
            ctx.set_source_rgba(get_rgb(index)[0], get_rgb(index)[1], get_rgb(index)[2], 1)
            ctx.fill()
        
        ctx.select_font_face("Helvetica", cairo.FONT_SLANT_NORMAL)
        ctx.set_font_size(vraster_size*.3)
        
        (x, y, width, height, dx, dy) = ctx.text_extents(str(np.ceil(mw*1000)/1000))
        cx = txr+padding+(24/24*len(matrix))*sn_width  + 1.75*padding + sn_width
        cy = yt-height/2


        ctx.move_to(cx - width/2, cy + height/4)    
        ctx.set_source_rgba(0, 0, 0, 1)
        ctx.show_text(str(np.ceil(mw*1000)/1000))
            
        mw = 0
        for t in range(len(times)):
            weights = matrix[t]
            weights = np.absolute(weights)

            m = np.sum(weights)/len(weights)
            if m>mw:
                mw = m

        if mw == 0:
            mw = 1

        self.avg_weights = []
        self.cols = []
        for t in range(len(times)):
            weights = matrix[t]
            weights = np.absolute(weights)

            self.avg_weights.append(np.mean(weights))

            weights = weights/mw

            if len(weights)==0:
                weights = np.array([0])

            

            adj_time = times[t]
            self.cols.append(adj_time)
    
    def update(self):
        self.r1.updatePops()
        self.r2.updatePops()

        self.draw()
        self.load_image()

    def generate_differences(self):
        # get the binned averages from each raster
        gn1 = self.r1.group_name
        gn2 = self.r2.group_name

        self.r1.calc_internal_var()
        self.r2.calc_internal_var()

        self.r1.group_name = gn1
        self.r2.group_name = gn2 
        self.r1.draw()
        self.r2.draw()

        pop1 = self.r1.pop_values.copy()
        pop2 = self.r2.pop_values.copy()

        times = list(pop1.keys())
        times.sort()

        running = []
        for g in gn1:
            self.r1.group_name = [g]
            self.r1.draw()

            m1 = self.r1.pop_values.copy()
            

            t1 = []
            t2 = []
            for t in times:
                t1.append(m1[t])
                t2.append(pop2[t])
            t1 = np.array(t1)
            t2 = np.array(t2)

            matrix = t2-t1
            diff = []
            for m in range(len(matrix)):
                diff.append([np.mean(np.absolute(matrix[m]))])

            columns = times
            difs = []
            for t in range(len(times)):
                beh = []
                for c in range(len(self.behaviors)):

                    ws = []
                    for c1 in range(len(self.behaviors)):
                        weight = (matrix[t,c*len(self.behaviors)+c1])
                        ws.append(np.absolute(weight))
                    beh.append(np.mean(np.array(ws)))
                difs.append(beh)

            diff = np.array(diff)
            difs = np.array(difs)

            total = np.concatenate((diff, difs), axis=1)
            total = np.transpose(total).tolist()

            running.extend(total)



        full_indices = []
        indices = ['total']
        indices.extend(self.behaviors)
        for g in gn1:
            full_indices.extend([i+' '+g for i in indices])


        df = pd.DataFrame(data=running, columns=columns, index=full_indices)

        df.to_csv(self.file1)

        running = []
        for g in gn2:
            self.r2.group_name = [g]
            self.r2.draw()

            m1 = self.r2.pop_values.copy()
            

            t1 = []
            t2 = []
            for t in times:
                t1.append(m1[t])
                t2.append(pop1[t])
            t1 = np.array(t1)
            t2 = np.array(t2)

            matrix = t2-t1
            diff = []
            for m in range(len(matrix)):
                diff.append([np.mean(np.absolute(matrix[m]))])

            columns = times
            difs = []
            for t in range(len(times)):
                beh = []
                for c in range(len(self.behaviors)):

                    ws = []
                    for c1 in range(len(self.behaviors)):
                        weight = (matrix[t,c*len(self.behaviors)+c1])
                        ws.append(np.absolute(weight))
                    beh.append(np.mean(np.array(ws)))
                difs.append(beh)

            diff = np.array(diff)
            difs = np.array(difs)

            total = np.concatenate((diff, difs), axis=1)
            total = np.transpose(total).tolist()

            running.extend(total)

        full_indices = []
        indices = ['total']
        indices.extend(self.behaviors)
        for g in gn2:
            full_indices.extend([i+' '+g for i in indices])


        df = pd.DataFrame(data=running, columns=columns, index=full_indices)

        df.to_csv(self.file2)

        
        self.r1.group_name = gn1
        self.r2.group_name = gn2 
        self.r1.draw()
        self.r2.draw()

class Ethograms:

    def __init__(self, window, window_title, behaviors, paths, groups, starts, lightcycles, maxdays, color_list=[(255,128,0),(225,192,0),(0,0,255),(255,0,0),(192,0,192),(153,87,238),(100,100,100),(0,192,0),(148,100,31)], width=800, height=800):
        self.window = window
        self.window.title(window_title)

        self.behaviors = behaviors

        self.width = width 
        self.height = height

        self.paths = paths

        self.groups = groups

        self.starts = starts

        self.lightcycles = lightcycles
        self.maxdays = maxdays

        self.file = 'C:\\Users\\Jones-Lab\\Documents\\cbas_test\\ethogram.png'

        self.timeseries(paths=paths, behaviors=behaviors, groups=groups, lightcycles=lightcycles)

        self.group_names.sort()
        self.group_name = [self.group_names[0]]



        self.frame = tk.Frame(self.window)

        self.right = tk.Frame(self.window)

        self.color_list = color_list


        self.canvas = tk.Canvas(self.frame, width=self.width, height=self.height)
        self.canvas.grid(column=0, row=0)

        
        self.frame.pack(side=tk.LEFT, pady=5, padx=5)

        
        options = self.group_names
        self.label = tk.Label(self.right, text = "Camera Groups", font=('TkDefaultFixed', 10)).pack(side=tk.TOP, anchor='sw', pady=5, padx=5)
        self.listbox = tk.Listbox(self.right, selectmode = "multiple")
        self.listbox.pack(pady=5, padx=5)
        
        for option in options:
            self.listbox.insert(tk.END, option)
        
        self.button = tk.Button(self.right, text="Replot", command=self.update, font=('TkDefaultFixed', 10), relief='flat', autostyle=False).pack(side=tk.BOTTOM, anchor='center', pady=5, padx=5)

        self.right.pack(side=tk.RIGHT, pady=5, padx=5)



        self.draw()
        self.load_image()

        
        self.window.mainloop()

    def draw(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.width, self.height)
        ctx = cairo.Context(surface)

        width = self.width
        height = self.height

        ctx.scale(width, height)
        
        self.draw_actograms(ctx, self.group_name)
        
        surface.write_to_png(self.file)

    def load_image(self):
        img = ImageTk.PhotoImage(Image.open(self.file))
        self.img = img

        self.canvas.create_image(0, 0, image=self.img, anchor=tk.NW)

    def draw_actograms(self, ctx, groups):

        print('Setting up all actograms...')

        num_behaviors = len(self.behaviors)
        num_wide = math.ceil(math.sqrt(num_behaviors))
        num_tall = math.ceil(num_behaviors/num_wide)

        padding = .02 

        actogram_width = (1 - 2*padding - (num_wide-1)*padding)/(num_wide)
        actogram_height = (1 - 2*padding - (num_tall-1)*padding)/(num_tall)

        cx = padding
        cy = padding
        i = 0

        for b in self.behaviors:
            h = int(i/num_wide)
            w = i%num_wide

            cx = (actogram_width)*w + (padding)*(w+1)
            cy = (actogram_height)*h + (padding)*(h+1)


            self.draw_actogram(ctx, cx, cy, actogram_width, actogram_height, padding, groups, b, self.color_list[self.behaviors.index(b)])
            i+=1

    def draw_actogram(self, ctx, tlx, tly, width, height, padding, groups, behavior, color, mode='light'):
        
        ctx.set_line_width(.005)

        if mode!='light':
            ctx.rectangle(tlx - padding/4,tly - padding/4,width + padding/2,height + padding/2)
            ctx.set_source_rgba(.1, .1, .1, 1)
            ctx.fill()
        else:
            ctx.rectangle(tlx - padding/4,tly - padding/4,width + padding/2,height + padding/2)
            ctx.set_source_rgba(1, 1, 1, 1)
            ctx.fill()


        print(f'Drawing {behavior} actogram...')
        opacity = 1/len(groups)
        
        for z in range(len(groups)):

            first = False 
            last = False 
            group = groups[z]

            if z==0:
                first = True
            if z==len(groups)-1:
                last = True

            total_data = []
            days = []
            cycles = []
            for p in range(len(self.timeseries_data[group])):
                data = self.timeseries_data[group][p]['data'][behavior]
                total_data.append(data)
                days.append(self.timeseries_data[group][p]['length'])
                cycles.append(self.timeseries_data[group][p]['light'])

            total_days = np.sum(days)
            if total_days>self.maxdays:
                total_days = self.maxdays


            day_height = height/total_days
            
            bin_width = 1/48 * self.bin_size/36000
            ds = 0
            for i in range(len(days)):
                
                data = total_data[:][i]

                times = data[0]
                ts = data[1]

                expdays = days[i]

                end = ds + expdays

                if end>self.maxdays:
                    end = self.maxdays
                
                for d in range(int(ds), int(end)):

                    by = tly+(d+1)*day_height

                    d1 = d-ds

                    valid = np.logical_and(times>(d1*24),times<=((d1+2)*24))

                    if d1%2!=0:
                        adj_times = times[valid] + 24
                    else:
                        adj_times = times[valid]

                    adj_times = adj_times%48

                    series = ts[valid]

                    # normalize the series
                    series = np.array(series)
                    series = series/np.max(series)

                    series = series*.90

                    LD = True
                    if cycles is not None:
                        LD = cycles[i]

                    if LD and first:
                        ctx.set_source_rgb(223/255,223/255,223/255)
                        ctx.rectangle(tlx+0/48*width, by-day_height, 6/48*width, day_height)
                        ctx.fill()
                        ctx.rectangle(tlx+18/48*width, by-day_height, 12/48*width, day_height)
                        ctx.fill()
                        ctx.rectangle(tlx+42/48*width, by-day_height, 6/48*width, day_height)
                        ctx.fill()

                        ctx.set_source_rgb(255/255, 239/255, 191/255)
                        ctx.rectangle(tlx+6/48*width, by-day_height, 12/48*width, day_height)
                        ctx.fill()
                        ctx.rectangle(tlx+30/48*width, by-day_height, 12/48*width, day_height)
                        ctx.fill()

                    elif first:
                        ctx.set_source_rgb(223/255,223/255,223/255)
                        ctx.rectangle(tlx+0/48*width, by-day_height, 6/48*width, day_height)
                        ctx.fill()
                        ctx.rectangle(tlx+18/48*width, by-day_height, 12/48*width, day_height)
                        ctx.fill()
                        ctx.rectangle(tlx+42/48*width, by-day_height, 6/48*width, day_height)
                        ctx.fill()

                        ctx.set_source_rgb(246/255, 246/255, 246/255)
                        ctx.rectangle(tlx+6/48*width, by-day_height, 12/48*width, day_height)
                        ctx.fill()
                        ctx.rectangle(tlx+30/48*width, by-day_height, 12/48*width, day_height)
                        ctx.fill()
                    


                    for t in range(len(adj_times)):
                        timepoint = adj_times[t]
                        value = series[t]
                        
                        a_time = timepoint/48
                        
                        ctx.rectangle(tlx+a_time*width,by-value*day_height,bin_width*width,value*day_height)
                        ctx.set_source_rgba(color[0]/255, color[1]/255, color[2]/255, opacity*2)
                        ctx.fill()


                ds += expdays

            if last:
                ds = 0
                for i in range(len(days)):

                    expdays = days[i]
                    end = ds + expdays

                    if end>self.maxdays:
                        end = self.maxdays
                    
                    for d in range(int(ds), int(ds+expdays)):

                        by = tly+(d+1)*day_height

                        ctx.set_line_width(.002)
                        ctx.set_source_rgb(0, 0, 0)
                        ctx.move_to(tlx, by)
                        ctx.line_to(tlx+48/48*width, by)
                        ctx.stroke()

                    
                    
                    ds += expdays
        
    def timeseries(self, paths, groups, lightcycles, behaviors):

        # paths is an array of paths to recording folders, it is assumed that the paths are in order of gluing
        
        # groups is an array of dictionaries of camera names for each recording that are to be glued together
        # [{0:'cam1', 1:'cam1'}] means that the first camera from the first recording path is going to be glued to the first camera of the second recording
        
        # lightcycles is a boolean array of whether the recording is in LD or not (True=LD, False=DD)
        # behaviors is an array of behaviors to plot

        print('Loading the timeseries data...')

        if groups is None:
            numpaths = len(paths)
            path = paths[0]  

            f = []
            for (dirpath, dirnames, filenames) in os.walk(path):
                f.extend(dirnames)
                break
            
            cameras = {}
            camera_names = []

            for file in f:
                try:
                    name = file.split('_')[1]
                except:
                    continue

                if name not in camera_names:
                    camera_names.append(name)
                    cameras[name] = []
            
            groups = [{p:c for p in range(numpaths)} for c in camera_names]


            

        # for path in paths:

        #     f = []
        #     for (dirpath, dirnames, filenames) in os.walk(path):
        #         f.extend(filenames)
        #         break
            
        #     cameras = {}
        #     camera_names = []

        #     for file in f:
        #         try:
        #             name = file.split('_')[1]
        #         except:
        #             continue

        #         if name in camera_names:
        #             cameras[name].append(file)
        #         else:
        #             camera_names.append(name)
        #             cameras[name] = [file]
        path = paths[0]   

        total_group_data = {}
        group_names = []

        ind = 0
        for arr in groups:
            total_group_data[ind] = None
            group_names.append(ind)
            ind+=1

        ind = 0
        for path in paths:


            exp = os.path.split(path)[1]

            f = []

            for (dirpath, dirnames, filenames) in os.walk(path):
                f.extend(dirnames)
                break


            deg = True 


            for fold in f:
                if os.path.exists(os.path.join(path, f[0], f[0]+'_outputs_inferences.csv')):
                    deg = False

            

            if not deg:
                files = {fold:os.path.join(path, fold, fold+'_outputs_inferences.csv') for fold in f}
            elif deg:
                files = {fold:os.path.join(path, fold, fold+'_predictions.csv') for fold in f}
            else:
                raise Exception('No valid files found in '+path)


            for file in files.keys():
                try:
                    name = file.split('_')[1]
                except:
                    continue

                if name in camera_names:
                    cameras[name].append(files[file])
                else:
                    camera_names.append(name)
                    cameras[name] = [files[file]]

            video_size = 18000
            bin_size = 18000
            self.bin_size = bin_size

            vidsperhour = 36000/video_size

            camera_data = {}
            timeseries_data = {}

            for g in camera_names:
                time_data = {}
                for file in cameras[g]:
                    df = pd.read_csv(file)
                    df = df.to_numpy()[1:,1:]

                    time_data[remove_leading_zeros(os.path.split(file)[1].split('_')[2])] = []

                    for r in range(len(df[0,:])):
                        bins = []
                        
                        for b in range(0,len(df),bin_size):
                            end = b+bin_size
                            if end>len(df):
                                end = len(df)
                            
                            col = df[b:end, r]
                            bins.append(np.sum(col))

                        bins = np.array(bins)
                        if len(bins)>video_size/bin_size:
                            bins = bins[:int(video_size/bin_size)]
                        elif len(bins)<video_size/bin_size:
                            n = video_size/bin_size - len(bins)
                            temp_bins = []
                            temp_bins.extend(bins.tolist())
                            for i in range(int(n)):
                                temp_bins.append(0)
                            
                            bins = np.array(temp_bins)

                        time_data[remove_leading_zeros(os.path.split(file)[1].split('_')[2])].append(bins)


                
                camera_data[g] = time_data

                num_bins = len(camera_data[camera_names[0]][0][0])
            
                days = int(np.ceil(len(camera_data[camera_names[0]])/(vidsperhour*24)))

                timeseries_data[g] = {b:None for b in behaviors}

                for i in np.arange(0,days,1):
                    
                    indices = range(int(i*num_bins*(vidsperhour*24)), int((i+1)*num_bins*(vidsperhour*24)),1)

                    valid_indices = []
                    for id in indices:
                        t = int(id)
                        tp = int(t/num_bins)
                        stp = int(t%num_bins)

                        if tp>=len(time_data):
                            continue
                        else:
                            valid_indices.append(id)
                    
                    indices = np.array(indices)

                    data = np.zeros((len(camera_data[camera_names[0]][0]), len(indices)))

                    time_data = camera_data[g]

                    for id in indices:
                        t = int(id)
                        tp = int(t/num_bins)
                        stp = int(t%num_bins)

                        if tp>=len(time_data):
                            continue

                        data_ind = int(id-i*num_bins*(vidsperhour*24))

                        for r in range(len(camera_data[camera_names[0]][0])):

                            try:
                                data[r,data_ind] = time_data[tp][r][stp]
                            except:
                                data[r,data_ind] = 0



                    for b in range(len(behaviors)):

                        ts = data[b, :]
                        times = np.arange(len(ts)) * (24) / ((video_size/bin_size) * (vidsperhour*24))
                        times += self.starts[ind]

                       
                        if timeseries_data[g][behaviors[b]] is None:
                            timeseries_data[g][behaviors[b]] = (times, ts)

                        else:
                            
                            old_data = timeseries_data[g][behaviors[b]]

                            newtimes = []
                            for t in old_data[0]:
                                newtimes.append(t)
                            for t in times:
                                newtimes.append(t+i*24)
                            newtimes = np.array(newtimes)

                            newts = []
                            for t in old_data[1]:
                                newts.append(t)
                            for t in ts:
                                newts.append(t)
                            newts = np.array(newts)


                            timeseries_data[g][behaviors[b]] = (newtimes, newts)

            recDays = int(np.ceil(len(camera_data[camera_names[0]])/(vidsperhour*24)))

            # loop over the groups
            for g in group_names:
                incl = groups[g]

                # get the camera for this group
                try:
                    recCam = incl[ind]
                except:
                    print(f'No data for recording {ind} and group {incl}.')
                    continue

                recStart = self.starts[ind]
                recLight = True
                if lightcycles is not None:
                    recLight = lightcycles[ind]

                if total_group_data[g] is None:
                    total_group_data[g] = [{'exp':os.path.split(path)[1],'length':recDays, 'start':recStart, 'light':recLight, 'data':timeseries_data[recCam]}]
                else:
                    total_group_data[g].append({'exp':os.path.split(path)[1],'length':recDays, 'start':recStart, 'light':recLight, 'data':timeseries_data[recCam]})
                

            # increase ind to new path
            ind+=1
            



        self.group_names = group_names

        self.timeseries_data = total_group_data

    def update(self):
        
        selections = self.listbox.curselection()

        if len(selections)==0:
            return
        
        self.group_name = [self.group_names[g] for g in selections]
        
        self.draw()
        self.load_image()
     
def remove_leading_zeros(num):
    for i in range(0,len(num)):
        if num[i]!='0':
            return int(num[i:])  
    return 0

def transitions(model_name, behaviors, recording_names, starts=None, name=None, project_config='undefined'):

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

        # grabbing the locations of the recordings
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

        # grabbing the locations of the recordings
        recordings_path = pconfig['recordings_path']


    data = []
    for r in recording_names:
        recording_path = os.path.join(recordings_path, r,model_name)
        recording_config = os.path.join(recordings_path, r, 'details.yaml')

        with open(recording_config, 'r') as file:
            rconfig = yaml.safe_load(file)
        
        if model_name in rconfig['cameras_per_model'].keys() and len(rconfig['cameras_per_model'][model_name])>0:
            try:
                data.append((recording_path, float(rconfig['start_time'].split(':')[0]) + float(rconfig['start_time'].split(':')[1])))
            except:
                data.append((recording_path, 0))


    if starts is None:
        starts = [data[i][1] for i in range(len(data))]

    if name is None:
        name = model_name+'_'+recording_names[0]

    paths = [data[i][0] for i in range(len(data))]


    root = tk.Tk()
    tg = TransitionRaster(root, 'Transition Raster', behaviors, paths, starts, name)

def transitiondifs(behaviors, recording_names, start1, start2, color1, color2, name1, name2):
    root = tk.Tk()
    tg = TransitionRasterDif(root, 'Transition Differences', behaviors, recording_names[0], recording_names[1], start1, start2, color1, color2, name1, name2)

def ethograms(model_name, behaviors, recording_names, starts=None, lightcycles=None, maxdays=50, groups=None, name=None, project_config='undefined'):

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

        # grabbing the locations of the recordings
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

        # grabbing the locations of the recordings
        recordings_path = pconfig['recordings_path']


    data = []
    for r in recording_names:
        recording_path = os.path.join(recordings_path, r,model_name)
        recording_config = os.path.join(recordings_path, r, 'details.yaml')

        with open(recording_config, 'r') as file:
            rconfig = yaml.safe_load(file)
        
        if model_name in rconfig['cameras_per_model'].keys() and len(rconfig['cameras_per_model'][model_name])>0:
            try:
                data.append((recording_path, float(rconfig['start_time'].split(':')[0]) + float(rconfig['start_time'].split(':')[1])))
            except:
                data.append((recording_path, 0))


    if starts is None:
        starts = [data[i][1] for i in range(len(data))]

    if name is None:
        name = model_name+'_'+recording_names[0]

    paths = [data[i][0] for i in range(len(data))]

    root = ttk.Window(themename=theme)
    tg = Ethograms(root, 'Ethograms', behaviors=behaviors, paths=paths, groups=groups, starts=starts, lightcycles=lightcycles, maxdays=maxdays)

##### UNDER CONSTRUCTION #####

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

def pop_fit_fold(path, behaviors, starttime, plot=False):

    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break


    groups = {}
    group_names = []

    for file in f:
        name = file.split('_')[1]

        if name in group_names:
            groups[name].append(file)
        else:
            group_names.append(name)
            groups[name] = [file]

    video_size = 18000
    bin_size = 6000

    fold = 96

    fold_nums = [1 for i in range(fold)]

    group_data = {}

    for g in group_names:
        time_data = {}
        for file in groups[g]:
            full_path = os.path.join(path, file)
            df = pd.read_csv(full_path)
            df = df.to_numpy()[1:,1:]

            try:

                data = time_data[remove_leading_zeros(file.split('_')[2])%fold]

                fold_nums[remove_leading_zeros(file.split('_')[2])%fold]+=1

                for r in range(len(df[0,:])):
                    bins = []
                    
                    for b in range(0,len(df),bin_size):
                        end = b+bin_size
                        if end>len(df):
                            end = len(df)
                        
                        col = df[b:end, r]
                        bins.append(np.sum(col))

                    bins = np.array(bins)
                    data[r] = data[r] + bins
                
                time_data[remove_leading_zeros(file.split('_')[2])%fold] = data

            except:
                time_data[remove_leading_zeros(file.split('_')[2])%fold] = []

                for r in range(len(df[0,:])):
                    bins = []
                    
                    for b in range(0,len(df),bin_size):
                        end = b+bin_size
                        if end>len(df):
                            end = len(df)
                        
                        col = df[b:end, r]
                        bins.append(np.sum(col))

                    bins = np.array(bins)
                    if len(bins)>video_size/bin_size:
                        bins = bins[:int(video_size/bin_size)]
                    elif len(bins)<video_size/bin_size:
                        n = video_size/bin_size - len(bins)
                        temp_bins = []
                        temp_bins.extend(bins.tolist())
                        for i in range(n):
                            temp_bins.append(0)
                        
                        bins = np.array(temp_bins)

                    time_data[remove_leading_zeros(file.split('_')[2])%fold].append(bins)
        
        group_data[g] = time_data

    num_bins = len(group_data[group_names[0]][0][0])
    
    data = np.zeros((len(group_data[group_names[0]][0]), len(group_data[group_names[0]][0][0])*fold*len(group_names)))
    
                

    col_names = []

    for i in range(num_bins*fold):
        for j,g in enumerate(group_names):
            col_names.append('T'+str(i)+'_rep'+str(j+1))
    

    for i,g in enumerate(group_names):
        indices = range(0, len(data[0]),len(group_names))

        time_data = group_data[g]

        for ind in indices:
            t = int(ind/len(group_names))
            tp = int(t/num_bins)
            stp = int(t%num_bins)


            for r in range(len(group_data[group_names[0]][0])):

                data[r,ind] = time_data[tp][r][stp]/fold_nums[tp]
    
    #print(data)
    df = pd.DataFrame(data, index=behaviors, columns=col_names)
    df.to_csv(os.path.join(path,"data\\data.csv"), sep='\t')


    df = file_parser.read_csv(os.path.join(path,"data\\data.csv"))
    #cosinor.periodogram_df(df)

    df_results = cosinor.population_fit_group(df, n_components = [3], period=144, plot=False) #folder=""

    models = population_models(df, n_components = [3], period=144)
    days = fold/48

    centroid_phases = []

    for b in behaviors:
        mx, my = models[b][0]
        mx = mx/(num_bins*24*2)*24
        my = my

        plt.plot(mx, my, color='black')

        # calculate the centroid

        dt = mx[1]-mx[0]

        # find the optimal time window
        mt = 0
        mv = 0
        for i in range(int(12/dt),len(mx)-int(12/dt)):
            start = int(i-12/dt)
            end = int(i+12/dt)

            if start<0:
                start = 0
            if end>len(mx):
                end = len(mx)

            avg = np.mean(my[start:end])
            if avg>mv:
                mt = i 
                mv = avg


        start = int(mt-12/dt)
        end = int(mt+12/dt)

        if start<0:
            start = 0
        if end>=len(mx):
            end = len(mx)-1
        
        period = np.logical_and(mx < mx[end],mx > mx[start])
        valid = my[period]

        q = np.quantile(valid, .75)
        overq = valid >= q

        dayvals = mx[period]
        plt.plot(dayvals[overq], q*np.ones(len(dayvals[overq])), '--', color='blue')
        
        w = np.multiply(dayvals[overq], valid[overq])
        m = np.mean(valid[overq])

        # Corrected calculation of the centroid
        cx = np.sum(w) / np.sum(valid[overq])
        centroid_phases.append((cx+starttime-6)%24)
        cy = m

        plt.plot(cx, cy, 'x', color='red')
        plt.title(b)
        if plot:
            plt.show()

    order = df_results['test'].to_numpy().tolist()

    acrophases = df_results['acrophase'].to_numpy()
    acrophases = acrophases*-1
    acrophases+=np.pi
    acrophases/=2*np.pi
    acrophases*=24
    acrophases-=12
    acrophases+=starttime 
    acrophases%=24
    acrophases-=6
    acrophases%=24


    df_results['acrophase'] = acrophases
    df_results['period'] = df_results['period'].to_numpy()/(24*num_bins*2)*24

    orders = []
    for b in behaviors:
        orders.append(order.index(b))
    df_results.insert(9, 'centroid_phase',pd.Series(centroid_phases, index=orders))

    df_results.to_csv(os.path.join(path,"results\\population_"+str(fold/2)+"hr_folded_results.csv"), index=False)

def fit_fold(path, behaviors, starttime, plot=False):

    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break


    groups = {}
    group_names = []

    for file in f:
        name = file.split('_')[1]

        if name in group_names:
            groups[name].append(file)
        else:
            group_names.append(name)
            groups[name] = [file]

    video_size = 18000
    bin_size = 6000

    fold = 96

    fold_nums = [1 for i in range(fold)]

    group_data = {}

    for g in group_names:
        time_data = {}
        for file in groups[g]:
            full_path = os.path.join(path, file)
            df = pd.read_csv(full_path)
            df = df.to_numpy()[1:,1:]

            try:

                data = time_data[remove_leading_zeros(file.split('_')[2])%fold]

                fold_nums[remove_leading_zeros(file.split('_')[2])%fold]+=1

                for r in range(len(df[0,:])):
                    bins = []
                    
                    for b in range(0,len(df),bin_size):
                        end = b+bin_size
                        if end>len(df):
                            end = len(df)
                        
                        col = df[b:end, r]
                        bins.append(np.sum(col))

                    bins = np.array(bins)
                    data[r] = data[r] + bins
                
                time_data[remove_leading_zeros(file.split('_')[2])%fold] = data

            except:
                time_data[remove_leading_zeros(file.split('_')[2])%fold] = []

                for r in range(len(df[0,:])):
                    bins = []
                    
                    for b in range(0,len(df),bin_size):
                        end = b+bin_size
                        if end>len(df):
                            end = len(df)
                        
                        col = df[b:end, r]
                        bins.append(np.sum(col))

                    bins = np.array(bins)
                    if len(bins)>video_size/bin_size:
                        bins = bins[:int(video_size/bin_size)]
                    elif len(bins)<video_size/bin_size:
                        n = video_size/bin_size - len(bins)
                        temp_bins = []
                        temp_bins.extend(bins.tolist())
                        for i in range(n):
                            temp_bins.append(0)
                        
                        bins = np.array(temp_bins)

                    time_data[remove_leading_zeros(file.split('_')[2])%fold].append(bins)
        
        group_data[g] = time_data

        num_bins = len(group_data[group_names[0]][0][0])
        
        data = np.zeros((len(group_data[group_names[0]][0]), num_bins*fold))
        
        col_names = []

        for i in range(num_bins*fold):
            col_names.append('T'+str(i)+'_rep'+str(1))
        

        indices = range(0, len(data[0]),1)
        time_data = group_data[g]

        for ind in indices:
            t = int(ind)
            tp = int(t/num_bins)
            stp = int(t%num_bins)


            for r in range(len(group_data[group_names[0]][0])):

                data[r,ind] = time_data[tp][r][stp]/fold_nums[tp]

        # Ok I can do rayleigh stats for each behavior in this form
        mags = []
        angles = []
        pvals = []
        periods = []
        powers = []
        faps = []
        for b in range(len(behaviors)):

            ts = data[b, :]
            times = np.arange(len(ts)) * (24) / (num_bins * 24 * 2)


            min_frequency = 1 / 28.0  # Minimum frequency (e.g., 28 hours)
            max_frequency = 1 / 20.0  # Maximum frequency (e.g., 20 hours)
            # Set the number of frequency points you want
            num_points = 1000  # Adjust as needed for desired resolution

            # Create a custom frequency grid
            custom_frequencies = np.linspace(1 / max_frequency, 1 / min_frequency, num_points)

            # Calculate the Lomb-Scargle periodogram for the custom frequencies
            power = LombScargle(times, ts).power(custom_frequencies)

            # Find the best-fitting period (frequency) and corresponding power
            powers.append(np.argmax(power))
            best_frequency = custom_frequencies[np.argmax(power)]
            best_period = best_frequency  # Invert the calculation
            periods.append(best_period)

            # Calculate the FAP for a desired significance level (e.g., alpha = 0.01 for 1% confidence)
            alpha = 0.01

            # Count the number of tested frequencies higher than or equal to the highest peak
            M = np.sum(power >= power[np.argmax(power)])

            # Calculate the FAP using the formula
            N = len(custom_frequencies)
            FAP = 1 - (1 - alpha)**(N / M)

            faps.append(1-FAP)





            times = np.arange(len(ts)) * (2 * np.pi) / (num_bins * 24 * 2)
            times = (times + starttime/24*2*np.pi) % (2 * np.pi)

            sumx = 0
            sumy = 0

            for time, val in enumerate(ts):
                angle = times[time]  # Use the time in radians directly
                sumx += np.cos(angle) * val
                sumy += np.sin(angle) * val


            mag = np.sqrt(sumx**2 + sumy**2)
            mags.append(mag)

            angle = math.atan2(sumy, sumx)

            # Convert the angle to hours (assuming 24-hour clock)
            angle = (angle / (2 * np.pi) * 24) % 24

            angles.append((angle-6)%24)

            pvals.append(circ_rtest(times, ts)[0])



        
        df = pd.DataFrame(data, index=behaviors, columns=col_names)
        df.to_csv(os.path.join(path,"data\\data.csv"), sep='\t')


        df = file_parser.read_csv(os.path.join(path,"data\\data.csv"))
        #cosinor.periodogram_df(df)

        df_results = cosinor.population_fit_group(df, n_components = [3], period=144, plot=False) #folder=""

        models = population_models(df, n_components = [3], period=144)
        days = fold/48

        centroid_phases = []

        for b in behaviors:
            mx, my = models[b][0]
            mx = mx/(num_bins*24*2)*24
            my = my

            plt.plot(mx, my, color='black')

            # calculate the centroid

            dt = mx[1]-mx[0]

            # find the optimal time window
            mt = 0
            mv = 0
            for i in range(int(12/dt),len(mx)-int(12/dt)):
                start = int(i-12/dt)
                end = int(i+12/dt)

                if start<0:
                    start = 0
                if end>len(mx):
                    end = len(mx)

                avg = np.mean(my[start:end])
                if avg>mv:
                    mt = i 
                    mv = avg


            start = int(mt-12/dt)
            end = int(mt+12/dt)

            if start<0:
                start = 0
            if end>=len(mx):
                end = len(mx)-1
            
            period = np.logical_and(mx < mx[end],mx > mx[start])
            valid = my[period]

            q = np.quantile(valid, .75)
            overq = valid >= q

            dayvals = mx[period]
            plt.plot(dayvals[overq], q*np.ones(len(dayvals[overq])), '--', color='blue')
            
            w = np.multiply(dayvals[overq], valid[overq])
            m = np.mean(valid[overq])

            # Corrected calculation of the centroid
            cx = np.sum(w) / np.sum(valid[overq])
            centroid_phases.append((cx+starttime-6)%24)
            cy = m

            plt.plot(cx, cy, 'x', color='red')
            plt.title(b)

            if plot:

                plt.show()

        order = df_results['test'].to_numpy().tolist()

        df_results['period'] = df_results['period'].to_numpy()/(24*num_bins*2)*24
        acrophases = df_results['acrophase'].to_numpy()
        acrophases = acrophases*-1
        acrophases+=np.pi
        acrophases/=2*np.pi
        acrophases*=24
        acrophases-=12
        acrophases+=starttime 
        acrophases%=24
        acrophases-=6
        acrophases%=24


        df_results['acrophase'] = acrophases

        orders = []
        for b in behaviors:
            orders.append(order.index(b))
        df_results.insert(9, 'centroid_phase',pd.Series(centroid_phases, index=orders))
        df_results.insert(9, 'rayleigh angle',pd.Series(angles, index=orders))
        df_results.insert(9, 'rayleigh mag',pd.Series(mags, index=orders))
        df_results.insert(9, 'rayleigh pval',pd.Series(pvals, index=orders))
        df_results.insert(9, 'lomb power',pd.Series(powers, index=orders))
        df_results.insert(9, 'lomb period',pd.Series(periods, index=orders))
        df_results.insert(9, 'lomb fap',pd.Series(faps, index=orders))
        df_results.to_csv(os.path.join(path,"results\\cam"+g+"_"+str(fold/2)+"hr_folded_results.csv"), index=False)

def fit_total(path, behaviors, starttime, plot=False, names=None):

    exp_name = os.path.split(path)[1]

    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break

    data = os.path.join(path, 'data')
    if not os.path.exists(data):
        os.mkdir(data)

    results = os.path.join(path, 'results')
    if not os.path.exists(results):
        os.mkdir(results)

    storage_folder = os.path.join(results,'total_recording')

    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)
    
    
    actograms = os.path.join(path,'actograms')

    if not os.path.exists(actograms):
        os.mkdir(actograms)


    groups = {}
    group_names = []

    for file in f:
        try:
            name = file.split('_')[1]
        except:
            continue

        if name in group_names:
            groups[name].append(file)
        else:
            group_names.append(name)
            groups[name] = [file]

    video_size = 18000
    bin_size = 18000

    group_data = {}

    for g in group_names:
        time_data = {}
        for file in groups[g]:
            full_path = os.path.join(path, file)
            df = pd.read_csv(full_path)
            df = df.to_numpy()[1:,1:]

            time_data[remove_leading_zeros(file.split('_')[2])] = []

            for r in range(len(df[0,:])):
                bins = []
                
                for b in range(0,len(df),bin_size):
                    end = b+bin_size
                    if end>len(df):
                        end = len(df)
                    
                    col = df[b:end, r]
                    bins.append(np.sum(col))

                bins = np.array(bins)
                if len(bins)>video_size/bin_size:
                    bins = bins[:int(video_size/bin_size)]
                elif len(bins)<video_size/bin_size:
                    n = video_size/bin_size - len(bins)
                    temp_bins = []
                    temp_bins.extend(bins.tolist())
                    for i in range(int(n)):
                        temp_bins.append(0)
                    
                    bins = np.array(temp_bins)

                time_data[remove_leading_zeros(file.split('_')[2])].append(bins)


        
        group_data[g] = time_data

        num_bins = len(group_data[group_names[0]][0][0])
        
        data = np.zeros((len(group_data[group_names[0]][0]), num_bins*len(group_data[group_names[0]])))
        
        col_names = []

        for i in range(num_bins*len(group_data[group_names[0]])):
            col_names.append('T'+str(i)+'_rep'+str(1))
        

        indices = range(0, len(data[0]),1)
        time_data = group_data[g]

        for ind in indices:
            t = int(ind)
            tp = int(t/num_bins)
            stp = int(t%num_bins)


            for r in range(len(group_data[group_names[0]][0])):

                try:
                    data[r,ind] = time_data[tp][r][stp]
                except:
                    data[r,ind] = 0


        # Ok I can do rayleigh stats for each behavior in this form
        mags = []
        angles = []
        pvals = []
        periods = []
        powers = []
        faps = []
        for b in range(len(behaviors)):

            ts = data[b, :]


            # make the actograms data frame
            delta = starttime - int(starttime)
            delta = int(60*delta)
            sstring = str(int(starttime))
            if len(sstring)==1:
                sstring = '0'+sstring
            dstring = str(int(delta))
            if len(dstring)==1:
                dstring = '0'+dstring

            

            metadata = [exp_name+"_"+g+'_'+behaviors[b],'01-jan-2000',sstring+':'+dstring, str(4*int(bin_size/600)),0,g,0]

            metadata.extend(ts.tolist())

            acto = pd.DataFrame(metadata)
            acto.to_csv(os.path.join(actograms,exp_name+"_"+g+'_'+behaviors[b]+'.csv'), index=False, header=False)
            os.rename(os.path.join(actograms,exp_name+"_"+g+'_'+behaviors[b]+'.csv'), os.path.join(actograms,exp_name+"_"+g+'_'+behaviors[b]+'.awd'))
            times = np.arange(len(ts)) * (24) / (num_bins * 24 * 2)


            min_frequency = 1 / 28.0  # Minimum frequency (e.g., 28 hours)
            max_frequency = 1 / 20.0  # Maximum frequency (e.g., 20 hours)
            # Set the number of frequency points you want
            num_points = 1000  # Adjust as needed for desired resolution

            # Create a custom frequency grid
            custom_frequencies = np.linspace(1 / max_frequency, 1 / min_frequency, num_points)

            # Calculate the Lomb-Scargle periodogram for the custom frequencies
            power = LombScargle(times, ts).power(custom_frequencies)

            # Find the best-fitting period (frequency) and corresponding power
            powers.append(np.argmax(power))
            best_frequency = custom_frequencies[np.argmax(power)]
            best_period = best_frequency  # Invert the calculation
            periods.append(best_period)

            # Calculate the FAP for a desired significance level (e.g., alpha = 0.01 for 1% confidence)
            alpha = 0.01

            # Count the number of tested frequencies higher than or equal to the highest peak
            M = np.sum(power >= power[np.argmax(power)])

            # Calculate the FAP using the formula
            N = len(custom_frequencies)
            FAP = 1 - (1 - alpha)**(N / M)

            faps.append(1-FAP)





            times = np.arange(len(ts)) * (2 * np.pi) / (num_bins * 24 * 2)
            times = (times + starttime/24*2*np.pi) % (2 * np.pi)

            sumx = 0
            sumy = 0

            for time, val in enumerate(ts):
                angle = times[time]  # Use the time in radians directly
                sumx += np.cos(angle) * val
                sumy += np.sin(angle) * val


            mag = np.sqrt(sumx**2 + sumy**2)
            mags.append(mag)

            angle = math.atan2(sumy, sumx)

            # Convert the angle to hours (assuming 24-hour clock)
            angle = (angle / (2 * np.pi) * 24) % 24

            angles.append((angle-6)%24)

            pvals.append(circ_rtest(times, ts)[0])
        
        
        df = pd.DataFrame(data, index=behaviors, columns=col_names)
        df.to_csv(os.path.join(path,"data\\data.csv"), sep='\t')


        df = file_parser.read_csv(os.path.join(path,"data\\data.csv"))
        #cosinor.periodogram_df(df)

        df_results = cosinor.population_fit_group(df, n_components = [3], period=144, plot=False) #folder=""

        models = population_models(df, n_components = [3], period=144)
        days = len(group_data[group_names[0]])/48

        centroid_phases = []

        for b in behaviors:
            mx, my = models[b][0]
            mx = mx/(num_bins*24*2)*24
            my = my

            if plot:
                plt.plot(mx, my, color='black')

            # calculate the centroid

            dt = mx[1]-mx[0]

            # find the optimal time window
            mt = 0
            mv = 0
            for i in range(int(6/dt),len(mx)-int(6/dt)):
                start = int(i-6/dt)
                end = int(i+6/dt)

                if start<0:
                    start = 0
                if end>len(mx):
                    end = len(mx)

                avg = np.mean(my[start:end])
                if avg>mv:
                    mt = i 
                    mv = avg


            start = int(mt-6/dt)
            end = int(mt+6/dt)

            if start<0:
                start = 0
            if end>=len(mx):
                end = len(mx)-1
            
            period = np.logical_and(mx < mx[end],mx > mx[start])
            valid = my[period]

            q = np.quantile(valid, .75)
            overq = valid >= q

            dayvals = mx[period]
            if plot:
                plt.plot(dayvals[overq], q*np.ones(len(dayvals[overq])), '--', color='blue')
            
            w = np.multiply(dayvals[overq], valid[overq])
            m = np.mean(valid[overq])

            # Corrected calculation of the centroid
            cx = np.sum(w) / np.sum(valid[overq])
            centroid_phases.append((cx+starttime-6)%24)
            cy = m

            if plot:
                plt.plot(cx, cy, 'x', color='red')
                plt.title(b)

            if plot:

                plt.show()

        order = df_results['test'].to_numpy().tolist()

        df_results['period'] = df_results['period'].to_numpy()/(24*num_bins*2)*24
        acrophases = df_results['acrophase'].to_numpy()
        acrophases = acrophases*-1
        acrophases+=np.pi
        acrophases/=2*np.pi
        acrophases*=24
        acrophases-=12
        acrophases+=starttime 
        acrophases%=24
        acrophases-=6
        acrophases%=24


        df_results['acrophase'] = acrophases

        orders = []
        for b in behaviors:
            orders.append(order.index(b))
        df_results.insert(9, 'centroid_phase',pd.Series(centroid_phases, index=orders))
        df_results.insert(9, 'rayleigh angle',pd.Series(angles, index=orders))
        df_results.insert(9, 'rayleigh mag',pd.Series(mags, index=orders))
        df_results.insert(9, 'rayleigh pval',pd.Series(pvals, index=orders))
        df_results.insert(9, 'lomb power',pd.Series(powers, index=orders))
        df_results.insert(9, 'lomb period',pd.Series(periods, index=orders))
        df_results.insert(9, 'lomb fap',pd.Series(faps, index=orders))

        df_results.to_csv(os.path.join(storage_folder,exp_name+"_"+g+"_stats.csv"), index=False)

def fit_daily(path, behaviors, starttime, plot=False):
    
    exp_name = os.path.split(path)[1]

    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break

    data = os.path.join(path, 'data')
    if not os.path.exists(data):
        os.mkdir(data)

    results = os.path.join(path, 'results')
    if not os.path.exists(results):
        os.mkdir(results)

    storage_folder = os.path.join(results,'daily')

    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)



    groups = {}
    group_names = []

    for file in f:
        try:
            name = file.split('_')[1]
        except:
            continue

        if name in group_names:
            groups[name].append(file)
        else:
            group_names.append(name)
            groups[name] = [file]

    video_size = 18000
    bin_size = 18000

    group_data = {}

    for g in group_names:
        time_data = {}
        for file in groups[g]:
            full_path = os.path.join(path, file)
            df = pd.read_csv(full_path)
            df = df.to_numpy()[1:,1:]

            time_data[remove_leading_zeros(file.split('_')[2])] = []

            for r in range(len(df[0,:])):
                bins = []
                
                for b in range(0,len(df),bin_size):
                    end = b+bin_size
                    if end>len(df):
                        end = len(df)
                    
                    col = df[b:end, r]
                    bins.append(np.sum(col))

                bins = np.array(bins)
                if len(bins)>video_size/bin_size:
                    bins = bins[:int(video_size/bin_size)]
                elif len(bins)<video_size/bin_size:
                    n = video_size/bin_size - len(bins)
                    temp_bins = []
                    temp_bins.extend(bins.tolist())
                    for i in range(int(n)):
                        temp_bins.append(0)
                    
                    bins = np.array(temp_bins)

                time_data[remove_leading_zeros(file.split('_')[2])].append(bins)
        
        group_data[g] = time_data

        num_bins = len(group_data[group_names[0]][0][0])
        
        
        days = int(np.ceil(len(group_data[group_names[0]])/48))

        df_total = []
        column_names = []

        for i in range(0,days,1):
            indices = range(i*num_bins*48, (i+1)*num_bins*48,1)

            valid_indices = []
            for ind in indices:
                t = int(ind)
                tp = int(t/num_bins)
                stp = int(t%num_bins)

                if tp>=len(time_data):
                    continue
                else:
                    valid_indices.append(ind)
            
            indices = np.array(indices)

            data = np.zeros((len(group_data[group_names[0]][0]), len(indices)))
        
            col_names = []

            for j in range(len(indices)):
                col_names.append('T'+str(j)+'_rep'+str(1))

            time_data = group_data[g]

            for ind in indices:
                t = int(ind)
                tp = int(t/num_bins)
                stp = int(t%num_bins)

                if tp>=len(time_data):
                    continue

                data_ind = int(ind-i*num_bins*48)

                for r in range(len(group_data[group_names[0]][0])):

                    
                    try:
                        data[r,data_ind] = time_data[tp][r][stp]
                    except:
                        data[r,data_ind] = 0
            
            
            # Ok I can do rayleigh stats for each behavior in this form
            mags = []
            angles = []
            pvals = []
            periods = []
            powers = []
            faps = []
            for b in range(len(behaviors)):

                ts = data[b, :]
                times = np.arange(len(ts)) * (24) / (num_bins * 24 * 2)


                min_frequency = 1 / 28.0  # Minimum frequency (e.g., 28 hours)
                max_frequency = 1 / 20.0  # Maximum frequency (e.g., 20 hours)
                # Set the number of frequency points you want
                num_points = 1000  # Adjust as needed for desired resolution

                # Create a custom frequency grid
                custom_frequencies = np.linspace(1 / max_frequency, 1 / min_frequency, num_points)

                # Calculate the Lomb-Scargle periodogram for the custom frequencies
                power = LombScargle(times, ts).power(custom_frequencies)

                # Find the best-fitting period (frequency) and corresponding power
                powers.append(np.argmax(power))
                best_frequency = custom_frequencies[np.argmax(power)]
                best_period = best_frequency  # Invert the calculation
                periods.append(best_period)

                # Calculate the FAP for a desired significance level (e.g., alpha = 0.01 for 1% confidence)
                alpha = 0.01

                # Count the number of tested frequencies higher than or equal to the highest peak
                M = np.sum(power >= power[np.argmax(power)])

                # Calculate the FAP using the formula
                N = len(custom_frequencies)
                FAP = 1 - (1 - alpha)**(N / M)

                faps.append(1-FAP)





                times = np.arange(len(ts)) * (2 * np.pi) / (num_bins * 24 * 2)
                times = (times + starttime/24*2*np.pi) % (2 * np.pi)

                sumx = 0
                sumy = 0

                for time, val in enumerate(ts):
                    angle = times[time]  # Use the time in radians directly
                    sumx += np.cos(angle) * val
                    sumy += np.sin(angle) * val


                mag = np.sqrt(sumx**2 + sumy**2)
                mags.append(mag)

                angle = math.atan2(sumy, sumx)

                # Convert the angle to hours (assuming 24-hour clock)
                angle = (angle / (2 * np.pi) * 24) % 24

                angles.append((angle-6)%24)

                pvals.append(circ_rtest(times, ts)[0])


            df = pd.DataFrame(data, index=behaviors, columns=col_names)
            df.to_csv(os.path.join(path,"data\\data.csv"), sep='\t')


            df = file_parser.read_csv(os.path.join(path,"data\\data.csv"))
            #cosinor.periodogram_df(df)

            df_results = cosinor.population_fit_group(df, n_components = [1], period=144, plot=False) #folder=""

            models = population_models(df, n_components = [1], period=144)
            days = 2

            centroid_phases = []

            for b in behaviors:
                mx, my = models[b][0]
                mx = mx/(num_bins*24*2)*24
                my = my

                if plot:
                    plt.plot(mx, my, color='black')

                # calculate the centroid

                dt = mx[1]-mx[0]

                # find the optimal time window
                mt = 0
                mv = 0
                for i in range(int(6/dt),len(mx)-int(6/dt)):
                    start = int(i-6/dt)
                    end = int(i+6/dt)

                    if start<0:
                        start = 0
                    if end>len(mx):
                        end = len(mx)

                    avg = np.mean(my[start:end])
                    if avg>mv:
                        mt = i 
                        mv = avg


                start = int(mt-6/dt)
                end = int(mt+6/dt)

                if start<0:
                    start = 0
                if end>=len(mx):
                    end = len(mx)-1
                
                period = np.logical_and(mx < mx[end],mx > mx[start])
                valid = my[period]

                q = np.quantile(valid, .75)
                overq = valid >= q

                dayvals = mx[period]
                if plot:
                    plt.plot(dayvals[overq], q*np.ones(len(dayvals[overq])), '--', color='blue')
                
                w = np.multiply(dayvals[overq], valid[overq])
                m = np.mean(valid[overq])

                # Corrected calculation of the centroid
                cx = np.sum(w) / np.sum(valid[overq])
                centroid_phases.append((cx+starttime-6)%24)
                cy = m

                if plot:
                    plt.plot(cx, cy, 'x', color='red')
                    plt.title(b)

                if plot:

                    plt.show()

            order = df_results['test'].to_numpy().tolist()

            df_results['period'] = df_results['period'].to_numpy()/(24*num_bins*2)*24
            acrophases = df_results['acrophase'].to_numpy()
            acrophases = acrophases*-1
            acrophases+=np.pi
            acrophases/=2*np.pi
            acrophases*=24
            acrophases-=12
            acrophases+=starttime 
            acrophases%=24
            acrophases-=6
            acrophases%=24


            df_results['acrophase'] = acrophases

            orders = []
            for b in behaviors:
                orders.append(order.index(b))
            df_results.insert(9, 'centroid_phase',pd.Series(centroid_phases, index=orders))
            df_results.insert(9, 'rayleigh angle',pd.Series(angles, index=orders))
            df_results.insert(9, 'rayleigh mag',pd.Series(mags, index=orders))
            df_results.insert(9, 'rayleigh pval',pd.Series(pvals, index=orders))
            df_results.insert(9, 'lomb power',pd.Series(powers, index=orders))
            df_results.insert(9, 'lomb period',pd.Series(periods, index=orders))
            df_results.insert(9, 'lomb fap',pd.Series(faps, index=orders))
            if len(df_total)==0:
                column_names = df_results.columns.to_numpy()
                df_total = df_results.to_numpy()
            else:
                df_total = np.concatenate((df_total,np.array([['' for c in range(len(column_names))]]),df_results.to_numpy()), axis=0)

        df_total = pd.DataFrame(df_total, columns=column_names)
        df_total.to_csv(os.path.join(storage_folder,exp_name+"_"+g+"_stats.csv"), index=False)

def fit_two_days(path, behaviors, starttime, plot=False):

    
    exp_name = os.path.split(path)[1]

    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break

    data = os.path.join(path, 'data')
    if not os.path.exists(data):
        os.mkdir(data)

    results = os.path.join(path, 'results')
    if not os.path.exists(results):
        os.mkdir(results)

    storage_folder = os.path.join(results,'two_days')

    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)


    groups = {}
    group_names = []

    for file in f:
        try:
            name = file.split('_')[1]
        except:
            continue

        if name in group_names:
            groups[name].append(file)
        else:
            group_names.append(name)
            groups[name] = [file]

    video_size = 18000
    bin_size = 18000

    group_data = {}

    for g in group_names:
        time_data = {}
        for file in groups[g]:
            full_path = os.path.join(path, file)
            df = pd.read_csv(full_path)
            df = df.to_numpy()[1:,1:]

            time_data[remove_leading_zeros(file.split('_')[2])] = []

            for r in range(len(df[0,:])):
                bins = []
                
                for b in range(0,len(df),bin_size):
                    end = b+bin_size
                    if end>len(df):
                        end = len(df)
                    
                    col = df[b:end, r]
                    bins.append(np.sum(col))

                bins = np.array(bins)
                if len(bins)>video_size/bin_size:
                    bins = bins[:int(video_size/bin_size)]
                elif len(bins)<video_size/bin_size:
                    n = video_size/bin_size - len(bins)
                    temp_bins = []
                    temp_bins.extend(bins.tolist())
                    for i in range(int(n)):
                        temp_bins.append(0)
                    
                    bins = np.array(temp_bins)

                time_data[remove_leading_zeros(file.split('_')[2])].append(bins)
        
        group_data[g] = time_data

        num_bins = len(group_data[group_names[0]][0][0])
        
        
        days = int(np.ceil(len(group_data[group_names[0]])/48))
        days_min = int(np.floor(len(group_data[group_names[0]])/48))

        df_total = []
        column_names = []

        for i in range(0,days-1,1):

            indices = range(i*num_bins*48, (i+2)*num_bins*48,1)

            valid_indices = []
            for ind in indices:
                t = int(ind)
                tp = int(t/num_bins)
                stp = int(t%num_bins)

                if tp>=len(time_data):
                    continue
                else:
                    valid_indices.append(ind)
            
            indices = np.array(indices)

            data = np.zeros((len(group_data[group_names[0]][0]), len(indices)))
        
            col_names = []

            for j in range(len(indices)):
                col_names.append('T'+str(j)+'_rep'+str(1))

            time_data = group_data[g]

            for ind in indices:
                t = int(ind)
                tp = int(t/num_bins)
                stp = int(t%num_bins)

                if tp>=len(time_data):
                    continue

                data_ind = int(ind-i*num_bins*48)

                for r in range(len(group_data[group_names[0]][0])):

                    try:
                        data[r,data_ind] = time_data[tp][r][stp]
                    except:
                        data[r,data_ind] = 0
            
            
            # Ok I can do rayleigh stats for each behavior in this form
            mags = []
            angles = []
            pvals = []
            periods = []
            powers = []
            faps = []
            for b in range(len(behaviors)):

                ts = data[b, :]
                times = np.arange(len(ts)) * (24) / (num_bins * 24 * 2)


                min_frequency = 1 / 28.0  # Minimum frequency (e.g., 28 hours)
                max_frequency = 1 / 20.0  # Maximum frequency (e.g., 20 hours)
                # Set the number of frequency points you want
                num_points = 1000  # Adjust as needed for desired resolution

                # Create a custom frequency grid
                custom_frequencies = np.linspace(1 / max_frequency, 1 / min_frequency, num_points)

                # Calculate the Lomb-Scargle periodogram for the custom frequencies
                power = LombScargle(times, ts).power(custom_frequencies)

                # Find the best-fitting period (frequency) and corresponding power
                powers.append(np.argmax(power))
                best_frequency = custom_frequencies[np.argmax(power)]
                best_period = best_frequency  # Invert the calculation
                periods.append(best_period)

                # Calculate the FAP for a desired significance level (e.g., alpha = 0.01 for 1% confidence)
                alpha = 0.01

                # Count the number of tested frequencies higher than or equal to the highest peak
                M = np.sum(power >= power[np.argmax(power)])

                # Calculate the FAP using the formula
                N = len(custom_frequencies)
                FAP = 1 - (1 - alpha)**(N / M)

                faps.append(1-FAP)





                times = np.arange(len(ts)) * (2 * np.pi) / (num_bins * 24 * 2)
                times = (times + starttime/24*2*np.pi) % (2 * np.pi)

                sumx = 0
                sumy = 0

                for time, val in enumerate(ts):
                    angle = times[time]  # Use the time in radians directly
                    sumx += np.cos(angle) * val
                    sumy += np.sin(angle) * val


                mag = np.sqrt(sumx**2 + sumy**2)
                mags.append(mag)

                angle = math.atan2(sumy, sumx)

                # Convert the angle to hours (assuming 24-hour clock)
                angle = (angle / (2 * np.pi) * 24) % 24

                angles.append((angle-6)%24)

                pvals.append(circ_rtest(times, ts)[0])


            df = pd.DataFrame(data, index=behaviors, columns=col_names)
            df.to_csv(os.path.join(path,"data\\data.csv"), sep='\t')


            df = file_parser.read_csv(os.path.join(path,"data\\data.csv"))
            #cosinor.periodogram_df(df)

            df_results = cosinor.population_fit_group(df, n_components = [1], period=144, plot=False) #folder=""

            models = population_models(df, n_components = [1], period=144)
            days = 2

            centroid_phases = []

            for b in behaviors:
                mx, my = models[b][0]
                mx = mx/(num_bins*24*2)*24
                my = my

                if plot:
                    plt.plot(mx, my, color='black')

                # calculate the centroid

                dt = mx[1]-mx[0]

                # find the optimal time window
                mt = 0
                mv = 0
                for i in range(int(6/dt),len(mx)-int(6/dt)):
                    start = int(i-6/dt)
                    end = int(i+6/dt)

                    if start<0:
                        start = 0
                    if end>len(mx):
                        end = len(mx)

                    avg = np.mean(my[start:end])
                    if avg>mv:
                        mt = i 
                        mv = avg


                start = int(mt-6/dt)
                end = int(mt+6/dt)

                if start<0:
                    start = 0
                if end>=len(mx):
                    end = len(mx)-1
                
                period = np.logical_and(mx < mx[end],mx > mx[start])
                valid = my[period]

                q = np.quantile(valid, .75)
                overq = valid >= q

                dayvals = mx[period]
                if plot:
                    plt.plot(dayvals[overq], q*np.ones(len(dayvals[overq])), '--', color='blue')
                
                w = np.multiply(dayvals[overq], valid[overq])
                m = np.mean(valid[overq])

                # Corrected calculation of the centroid
                cx = np.sum(w) / np.sum(valid[overq])
                centroid_phases.append((cx+starttime-6)%24)
                cy = m

                if plot:
                    plt.plot(cx, cy, 'x', color='red')
                    plt.title(b)

                if plot:

                    plt.show()

            order = df_results['test'].to_numpy().tolist()

            df_results['period'] = df_results['period'].to_numpy()/(24*num_bins*2)*24
            acrophases = df_results['acrophase'].to_numpy()
            acrophases = acrophases*-1
            acrophases+=np.pi
            acrophases/=2*np.pi
            acrophases*=24
            acrophases-=12
            acrophases+=starttime 
            acrophases%=24
            acrophases-=6
            acrophases%=24


            df_results['acrophase'] = acrophases

            orders = []
            for b in behaviors:
                orders.append(order.index(b))
            df_results.insert(9, 'centroid_phase',pd.Series(centroid_phases, index=orders))
            df_results.insert(9, 'rayleigh angle',pd.Series(angles, index=orders))
            df_results.insert(9, 'rayleigh mag',pd.Series(mags, index=orders))
            df_results.insert(9, 'rayleigh pval',pd.Series(pvals, index=orders))
            df_results.insert(9, 'lomb power',pd.Series(powers, index=orders))
            df_results.insert(9, 'lomb period',pd.Series(periods, index=orders))
            df_results.insert(9, 'lomb fap',pd.Series(faps, index=orders))
            if len(df_total)==0:
                column_names = df_results.columns.to_numpy()
                df_total = df_results.to_numpy()
            else:
                df_total = np.concatenate((df_total,np.array([['' for c in range(len(column_names))]]),df_results.to_numpy()), axis=0)

        df_total = pd.DataFrame(df_total, columns=column_names)
        df_total.to_csv(os.path.join(storage_folder,exp_name+"_"+g+"_stats.csv"), index=False)

def raw_transitions_over_time(behaviors, bin_size, jump_size, path):
    exp_name = os.path.split(path)[1]

    f = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        f.extend(filenames)
        break

    markov = os.path.join(path, 'markov')
    if not os.path.exists(markov):
        os.mkdir(markov)

    storage_folder = os.path.join(markov,f'{bin_size}_frame_bin_{jump_size}_bin_jump')

    if not os.path.exists(storage_folder):
        os.mkdir(storage_folder)

    groups = {}
    group_names = []

    for file in f:
        try:
            name = file.split('_')[1]
        except:
            continue

        if name in group_names:
            groups[name].append(file)
        else:
            group_names.append(name)
            groups[name] = [file]


    transition_data = {}

    col_names = []
    for b in behaviors:
        for b1 in behaviors:
            col_names.append(b+'->'+b1)


    for g in group_names:
        total = []

        files = groups[g]
        for file in files:

            df = pd.read_csv(os.path.join(path, file))
            df = df.to_numpy()[1:,1:]

            matrix = np.zeros((len(behaviors), len(behaviors)))

            linear = []

            for i in range(len(df)):
                linear.append(np.argmax(df[i]))
            
            # calc the transitions
            linear = np.array(linear)
            bin_size = bin_size

            jump_size = jump_size

            binned = []

            for i in range(0,len(linear),bin_size):

                end = i+bin_size
                if end>len(linear):
                    end = len(linear)

                chunk = linear[i:end]
                counts = []
                for b in range(len(behaviors)):
                    counts.append(np.sum(chunk==b))
                if counts[np.argmax(counts)]==0:
                    binned.append(-1)
                else:
                    binned.append(np.argmax(counts))
            
            instances = [0 for b in behaviors]
            for i in range(len(binned)):
                instances[binned[i]] += 1

            for b in range(len(behaviors)):
                next = [0 for b1 in behaviors]
                for i in range(len(binned)-jump_size):
                    if b==binned[i]:
                        next[binned[i+jump_size]] += 1
                next = np.array(next)
                if instances[b]!=0:
                    next = next/instances[b]

                matrix[b,:] = next
            
            total.append(matrix.flatten())
    
        transition_data[g] = np.array(total)

        file_name = os.path.join(storage_folder, exp_name+'_'+g+'_transitions.csv')
        df = pd.DataFrame(np.array(total),columns=col_names)
        df.to_csv(file_name, index=False)
     
def calc_all_data(paths, behaviors, starttimes, bins, jumps):        
    
    for path, starttime in zip(paths,starttimes):
        fit_total(path, behaviors, starttime, False)
        fit_daily(path, behaviors, starttime, False)
        fit_two_days(path, behaviors, starttime, False)
    
def population_fit(df_pop, n_components = 2, period = 24, lin_comp= False, model_type = 'lin', 
                   plot = True, plot_measurements=True, plot_individuals=True, plot_margins=True, hold = False, save_to = '', x_label='', y_label='', 
                   return_individual_params = False, params_CI = False, samples_per_param_CI=5, max_samples_CI = 1000, 
                   sampling_type = "LHS", parameters_to_analyse = ['amplitude', 'acrophase', 'mesor'], parameters_angular = ['acrophase'], color="black", **kwargs):

    #if return_individual_params:
    ind_params = {}
    ind_params_stats = {}
    for param in parameters_to_analyse:
        ind_params[param] = []

        
    params = -1

    tests = df_pop.test.unique()
    k = len(tests)
    

    #X_test = np.linspace(0, 2*period, 1000)
    #X_fit_eval_params = generate_independents(X_test, n_components = n_components, period = period, lin_comp = lin_comp)
    #if lin_comp:
    #    X_fit_eval_params[:,1] = 0    
    
    min_X = np.min(df_pop.x.values)
    max_X = np.max(df_pop.x.values)
    min_Y = np.min(df_pop.y.values)
    max_Y = np.max(df_pop.y.values)


    if plot:
        if plot_measurements:
            X_plot = np.linspace(min(min_X,0), 1.1*max(max_X,period), 1000)
        else:
            X_plot = np.linspace(0, 1.1*period, 1000)

        X_plot_fits = cosinor.generate_independents(X_plot, n_components = n_components, period = period, lin_comp = lin_comp)
        #if lin_comp:
        #    X_plot_fits[:,1] = 0   

    """
    min_X = 1000
    max_X = 0
    min_Y = 1000
    max_Y = 0
    min_X_test = np.min(X_test)
    """
    min_Y_test = 1000
    max_Y_test = 0

    models = []
    
    
    for test in tests:
        x,y = df_pop[df_pop.test == test].x.values, df_pop[df_pop.test == test].y.values
        
        """
        min_X = min(min_X, np.min(x))
        max_X = max(max_X, np.max(x))
        
        min_Y = min(min_Y, np.min(y))
        max_Y = max(max_Y, np.max(y))
        """

        results, statistics, rhythm_params, X_test, Y_test, model = cosinor.fit_me(x, y, n_components = n_components, period = period, plot = False, return_model = True, lin_comp=lin_comp, **kwargs)
        
        Y_plot_fits = results.predict(X_plot_fits) 

        # plt.plot(X_plot, Y_plot_fits)
        # plt.show()

        models.append((X_plot, Y_plot_fits))

    return models

def population_models(df, n_components = 2, period = 24, folder = '', prefix='', names = [], **kwargs):

    if type(period) == int:
        period = [period]
        
    if type(n_components) == int:
        n_components = [n_components]
        

    if not any(names):
        names = np.unique(df.test) 

    names = list(set(list(map(lambda x:x.split('_rep')[0], names))))
    names.sort()
    
    models_test = {}
    
    for name in set(names):
        for n_comps in n_components:
            for per in period:            
                if n_comps == 0:
                    per = 100000
                    
                    
                df_pop = df[df.test.str.startswith(name)]   

                if folder:                                       
                    models = population_fit(df_pop, n_components = n_comps, period = per, **kwargs)
                else:                    
                    models = population_fit(df_pop, n_components = n_comps, period = per, **kwargs)
                
                models_test[name] = models
    
    return models_test