
import os
import glob
import time
import traceback
import subprocess
import threading
import matplotlib
import matplotlib.pyplot as plt
import tkinter
matplotlib.use('TkAgg')
import numpy as np
import pandas as pd
import cv2
from queue import Queue
from typing import Union
import sys
from omegaconf import DictConfig, OmegaConf
import math

from deepethogram import projects, utils, configuration
from deepethogram.postprocessing import get_postprocessor_from_cfg
from datetime import datetime
import PyQt5

import sklearn.metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tabulate import tabulate
import umap.umap_ as umap


import joblib
# from functools import partial
# import shutil

# import numpy as np
# from omegaconf import ListConfig
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# from deepethogram.configuration import make_feature_extractor_inference_cfg
# from deepethogram import projects, utils
# from deepethogram.data.augs import get_cpu_transforms, get_gpu_transforms
# from deepethogram.data.datasets import VideoIterable
# from deepethogram.flow_generator.train import build_model_from_cfg as build_flow_generator
# from deepethogram.flow_generator.utils import flow_to_rgb_polar, flow_to_rgb

import deepethogram.feature_extractor.inference as fei



class headless_deg:

    def __init__(self, project_head: Union[str, os.PathLike]):
        self.cfg = None 
        self.data_path = None 
        self.model_path = None
        self.default_archs = None
        self.trained_model_dict = None
        self.labelfile = None
        self.outputfile = None 
        self.videofile = None
        self.postprocessor = None
        self.probabilities = None
        self.estimated_labels = None
        self.thresholds = None
        self.latent_name = None
        
        self.built_fe = False
        self.fe_cfg = None

        self.inference_pipe = None
        self.listener = None

        self.initialize_project(project_head)
        
        self.diction = {'cam 1':[],'cam 2':[],'cam 3':[],'cam 4':[],'cam 5':[],'cam 6':[],'cam 7':[],'cam 8':[],'cam 9':[],'cam 10':[],'cam 11':[],'cam 12':[],'cam 13':[],'cam 14':[],'cam 15':[],'cam 16':[]}
        
        self.motion_means = {'background':[0,0],'eating':[0,0],'drinking':[0,0],'rearing':[0,0],'climbing':[0,0],'digging':[0,0],'nesting':[0,0],'resting':[0,0],'grooming':[0,0]}
        
        self.motion_var = {'background':[0,0],'eating':[0,0],'drinking':[0,0],'rearing':[0,0],'climbing':[0,0],'digging':[0,0],'nesting':[0,0],'resting':[0,0],'grooming':[0,0]}


    def initialize_project(self, directory: Union[str, os.PathLike]):

        if len(directory) == 0:
            return
        filename = os.path.join(directory, 'project_config.yaml')

        if len(filename) == 0 or not os.path.isfile(filename):
            print('something wrong with loading yaml file: {}'.format(filename))
            return

        # overwrite cfg passed at command line now that we know the project path. still includes command line arguments
        self.cfg = configuration.make_config(directory, ['config', 'gui', 'postprocessor'], run_type='gui', model=None)


        self.cfg = projects.convert_config_paths_to_absolute(self.cfg, raise_error_if_pretrained_missing=False)


        self.cfg = projects.setup_run(self.cfg, raise_error_if_pretrained_missing=False)


        # for convenience
        self.data_path = self.cfg.project.data_path
        self.model_path = self.cfg.project.model_path

        self.get_trained_models()


    def get_trained_models(self):
        trained_models = projects.get_weights_from_model_path(self.model_path)
        self.get_default_archs()
        trained_dict = {}

        self.trained_model_dict = trained_dict

        for model, archs in trained_models.items():
            trained_dict[model] = {}

            # for sequence models, we can train with no pre-trained weights
            if model == 'sequence':
                trained_dict[model][''] = None

            arch = self.default_archs[model]['arch']
            if arch not in archs.keys():
                continue
            trained_dict[model]['no pretrained weights'] = None
            for run in trained_models[model][arch]:
                key = os.path.basename(os.path.dirname(run))
                if key == 'lightning_checkpoints':
                    key = os.path.basename(os.path.dirname(os.path.dirname(run)))
                trained_dict[model][key] = run


    # initializes the default architectures of the models
    def get_default_archs(self):
        
        if 'preset' in self.cfg:
            preset = self.cfg.preset
        else:
            preset = 'deg_f'
        default_archs = projects.load_default('preset/{}'.format(preset))
        seq_default = projects.load_default('model/sequence')
        default_archs['sequence'] = {'arch': seq_default['sequence']['arch']}

        if 'feature_extractor' in self.cfg and self.cfg.feature_extractor.arch is not None:
            default_archs['feature_extractor']['arch'] = self.cfg.feature_extractor.arch
        if 'flow_generator' in self.cfg and self.cfg.flow_generator.arch is not None:
            default_archs['flow_generator']['arch'] = self.cfg.flow_generator.arch
        if 'sequence' in self.cfg and 'arch' in self.cfg.sequence and self.cfg.sequence.arch is not None:
            default_archs['sequence']['arch'] = self.cfg.sequence.arch
        self.default_archs = default_archs

    def add_multiple_videos(self, filenames):
        if self.data_path is not None:
            data_dir = self.data_path
        else:
            raise ValueError('create or load a DEG project before loading video')


        if len(filenames) == 0:
            return
        for filename in filenames:
            assert os.path.exists(filename[0])

        for filename in filenames:
            self.initialize_video(filename[0])


    def initialize_video(self, videofile: Union[str, os.PathLike]):
        
        try: 

            if  os.path.normpath(self.cfg.project.data_path) in os.path.normpath(videofile) and projects.is_deg_file(videofile):

                record = projects.get_record_from_subdir(os.path.dirname(videofile))
                
                labelfile = record['label']
                outputfile = record['output']

                self.labelfile = labelfile
                self.outputfile = outputfile

                if labelfile is not None:
                    self.import_labelfile(labelfile)

                if outputfile is not None:
                    self.import_outputfile(outputfile, first_time=True)
            else:
            
                new_loc = projects.add_video_to_project(OmegaConf.to_container(self.cfg), videofile, mode='move')
                
                self.videofile = new_loc
                
                utils.load_yaml(os.path.join(os.path.dirname(self.videofile), 'record.yaml'))



        except BaseException as e:
            tb = traceback.format_exc()
            print(tb)
            return

    def import_labelfile(self, labelfile: Union[str, os.PathLike]):
        
        assert (os.path.isfile(labelfile))
        df = pd.read_csv(labelfile, index_col=0)
        array = df.values
    
    def roll(self, array, window):
        l = len(array)
        result = np.zeros(l)
        for i in range(0, l):
            s = 0
            n = 0
            for j in range(0, window):
                if i+j<l:
                    s+=array[i+j]
                    n+=1
            if n!=0:
                result[i] = s/n
        return result
    def match_shape(self, arr1, arr2):
        """
        Adjusts the shape of arr2 to match the shape of arr1.
        Pads with zeros or reduces the size of arr2 as necessary.
        """
        shape1 = np.array(arr1.shape)
        shape2 = np.array(arr2.shape)
        
        # If shapes are already the same
        if np.array_equal(shape1, shape2):
            return arr2
        
        # Otherwise, adjust arr2's shape
        slices = tuple(slice(0, min(dim1, dim2)) for dim1, dim2 in zip(shape1, shape2))
        result = arr2[slices].copy()

        # Pad the result if necessary
        pad_width = [(0, dim1 - min(dim1, dim2)) for dim1, dim2 in zip(shape1, shape2)]
        result = np.pad(result, pad_width, mode='constant')

        return result
        

    def import_outputfile(self, outputfile: Union[str, os.PathLike], latent_name=None, first_time: bool = False):

        if outputfile is None:
            return
        try:
            outputs = projects.import_outputfile(self.cfg.project.path,
                                                 outputfile,
                                                 class_names=OmegaConf.to_container(self.cfg.project.class_names),
                                                 latent_name='resnet50')
        except ValueError as e:
            print('If you got a broadcasting error: did you add or remove behaviors and not re-train?')
            return

        probabilities, thresholds, latent_name, keys = outputs

        #thresholds[5] -= .21
        path, filename = os.path.split(outputfile)
        print(filename)
        fn = filename.split('_outputs')[0]

        fn = filename.split('_outputs')[0]

        
        #fn_labels = os.path.join(path, fn+'_labels.csv')

        #df_labels = pd.read_csv(fn_labels)

        # i = 0
        # behs = {'background':0,'eating':1, 'drinking':2, 'rearing':3, 'climbing':4, 'digging':5, 'nesting':6, 'resting':7, 'grooming':8}
        # behaviors = ['background','eating', 'drinking', 'rearing', 'climbing', 'digging', 'nesting', 'resting', 'grooming']
        # for b in behaviors:
        #     #fpr, tpr, thres = sklearn.metrics.roc_curve(df_labels[b].values, probabilities[:,behs[b]],pos_label=1)
        #     precision, recall, thres = sklearn.metrics.precision_recall_curve(df_labels[b].values, probabilities[:,behs[b]])

        #     fscore = 2*precision*recall/(precision+recall)
        #     # calculate the g-mean for each threshold
        #     #gmeans = np.sqrt(tpr * (1-fpr))
        #     # locate the index of the largest g-mean
        #     ix = np.argmax(fscore)
        #     print(b)
        #     print(fscore[ix])
        #     print(thres[ix])
        #     thresholds[i] = thres[ix]
        #     #print(sklearn.metrics.auc(fpr,tpr))
        #     i+=1
        #thresholds[8] = 0.32
        #self.postprocessor = get_postprocessor_from_cfg(self.cfg, thresholds)

        #####

        path, filename = os.path.split(outputfile)
        fn = filename.split('_outputs')[0]
        file = fn
        vid = os.path.join(path, fn+'.mp4')
        probfile = os.path.join(path, fn+'_probs.csv')

        cap = cv2.VideoCapture(vid)

    
        ret, frame1 = cap.read()
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        hsv = np.zeros_like(frame1)
        hsv[...,1] = 255
                
        motion_values = []
        positions = []
        fn = 0
        while(1):
            if fn%10!=0:
                ret, frame2 = cap.read()
                if not ret:
                    break
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                fn+=1
                continue
            ret, frame2 = cap.read()
            if not ret:
                break
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

            # Calculate dense optical flow using Farneback method
            flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            
            # Compute magnitude and angle of 2D vectors
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

            # Set hue according to the optical flow direction
            hsv[...,0] = ang*180/np.pi/2

            # Set value according to the optical flow magnitude (normalized)
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            # Convert HSV to BGR
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            # For a scalar quantity of motion, you can sum the magnitudes
            motion_value = np.sum(mag)
            for i in range(0,10):
                motion_values.append(motion_value)
            
            # Update the previous frame for the next iteration
            prvs = next
            fn+=1


        cap.release()



        derivs = np.copy(probabilities)
        motion_values = np.array(motion_values)
        motion_values[np.isnan(motion_values)] = 0
        motion_values[np.isinf(motion_values)] = 10000
        for i in range(0,9):
            derivs[:,i] = self.match_shape(derivs[:,i],motion_values)
        

        
        #try:
        smoothed_cnn = np.copy(probabilities)
        for i in range(0,len(probabilities[0,:])):
            smoothed_cnn[:,i] = self.roll(probabilities[:,i],10)


        diff_outputs = np.diff(smoothed_cnn, axis=0)
        # Append a row at the beginning or end to make the shape consistent
        diff_outputs = np.vstack([np.zeros(diff_outputs.shape[1]), diff_outputs])

        #extinction feature
        extinction_cnn = np.copy(probabilities)
        for i in range(0,len(probabilities[0,:])):
            extinction_cnn[:,i] = np.where(probabilities[:,i]>thresholds[i],1,0)

        for i in range(0,len(probabilities[0,:])):
            s = 0
            for fn in range(0,len(probabilities[:,i])):
                if extinction_cnn[fn,i]==1:
                    s+=1
                else:
                    s=0
                
                if s<0:
                    s = 0
                extinction_cnn[fn,i] = s

        features = np.hstack([probabilities, diff_outputs, extinction_cnn, derivs])

        

        #### RF POST PROCESSOR
        loaded_model = joblib.load('C:\\Users\\Jones-Lab\\Documents\DEG\\random_forest_model_motion_filter.sav')
        try:
            result1 = loaded_model.predict(features)
        except:
            print("RTF model failed to predict")

        probs = loaded_model.predict_proba(features)
        top3_classes = np.argsort(-probs, axis=1)[:, :3]
        print(top3_classes)

        classes = [0,0,0,0,0,0,0,0,0]

        for i in range(0,len(result1)):
            if result1[i]==1:

                if top3_classes[i,1]==8:
                    classes[int(result1[i])-1] += 1
                    result1[i] = 8
            if result1[i]==9 and probs[i,int(result1[i])]<.9:

                if top3_classes[i,1]==8:
                    classes[int(result1[i])-1] += 1
                    result1[i] = 8

            # if motion_values[i]<1000 and (result1[i]==9 or result1[i]==1):
            #     classes[int(result1[i])-1] += 1
            #     result1[i] = 8

        print(classes)

        behaviors = ['motion','background','eating', 'drinking', 'rearing', 'climbing', 'digging', 'nesting', 'resting', 'grooming']
        ndf = pd.DataFrame(probs, columns=behaviors)

        #motion_values = np.hstack([np.zeros(1), motion_values])
        ndf['motion'] = self.match_shape(np.array(ndf['motion'].values),motion_values)

        ndf.to_csv(probfile)

        PCA = False 
        if PCA:
            fn_labels = os.path.join(path, file+'_labels.csv')
            df_labels = pd.read_csv(fn_labels)
            labels = df_labels.to_numpy()

            data = probs
            reducer = umap.UMAP(verbose=True)
            embedding = reducer.fit_transform(data)

            actual_vals = np.zeros(len(labels[:,0]))

            for fn in range(0,len(labels[:,0])):
                row = labels[fn]
                for v in range(0,len(row)):
                    if row[v]==1:
                        actual_vals[fn] = v 
                        break
            
            behavior_colors = ['black','black','red','green','sienna','goldenrod','purple','deeppink','blue','teal']
            labels = [behavior_colors[int(v)] for v in actual_vals]


            plt.scatter(embedding[:,0],embedding[:,1], c=labels, cmap='Spectral',s=1)
            plt.colorbar(boundaries=np.arange(10)-.5).set_ticks(np.arange(9))
            plt.savefig('embedding.png')



        # loaded_model = joblib.load('C:\\Users\\Jones-Lab\\Documents\DEG\\random_forest_model_resting_bg.sav')
        # try:
        #     result2 = loaded_model.predict(features)
        # except:
        #     print("RTF model failed to predict")

        estimated_labels = np.zeros(probabilities.shape)
        i = 0
        for v in result1:
            estimated_labels[i,int(v-1)] = 1
            i+=1
        i = 0

        #####

        # for v in result2:
        #     if estimated_labels[i,7]==0 and v==8:
        #         estimated_labels[i,7] = 1
        #         estimated_labels[i,0] = 0
        #     i+=1
        #except:
        #    print("Post processor failure")

        ### BASIC POST PROCESSOR
        #self.postprocessor = get_postprocessor_from_cfg(self.cfg, thresholds)
        #estimated_labels = self.postprocessor(probabilities)

        
        #out_nest_correlation = {'background':1,'eating':.9, 'drinking':.9, 'rearing':.5, 'climbing':1, 'digging':1, 'nesting':-.5, 'resting':-1, 'grooming':-.2}
        
        #motion = probabilities[:,0]*2 + probabilities[:,1]*2 + probabilities[:,2]*2 + probabilities[:,3]*2 + probabilities[:,4]*2 + probabilities[:,5]*2 + probabilities[:,6] + probabilities[:,8]
        #motion = motion/14
        #estimated_labels = self.postprocessor(probabilities)
        # for b in behaviors:
        #     plt.clf()
        #     fn = filename.split('_outputs')[0]
        #     fn = os.path.join(path, fn+'_'+b+'.png')
        #     rolled = probabilities[:,behs[b]]

        #     fn_pic = os.path.join(path, fn)

        #     plt.plot(np.arange(0,len(probabilities)), probabilities[:,behs[b]])
        #     plt.plot(np.arange(0,len(probabilities)), df_labels[b].values)

            
            
        #     threshold = np.ones(len(probabilities))*thresholds[behs[b]]


        #     if b =='resting':
        #         motion = (thresholds[0] - self.roll(probabilities[:,0], 500))*-1
        #         threshold += motion
        #         threshold += self.std_win(probabilities[:,7],50)**(1/2)
        #         rolled[rolled>threshold] = 1
        #         rolled[rolled<threshold] = 0
        #         estimated_labels[:,behs[b]] = rolled
        #     elif b =='digging':
        #         #motion = (thresholds[0] - self.roll(probabilities[:,0], 500))*1
        #         #threshold += motion
        #         threshold -= self.std_win(probabilities[:,behs[b]],50)*2
        #         rolled[rolled>threshold] = 1
        #         rolled[rolled<threshold] = 0
        #         estimated_labels[:,behs[b]] = rolled
        #     elif b!='background':
        #         #motion = (thresholds[0] - self.roll(probabilities[:,0], 500))*1
        #         #threshold += motion
        #         threshold -= self.std_win(probabilities[:,behs[b]],50)*2
        #         rolled[rolled>threshold] = 1
        #         rolled[rolled<threshold] = 0
        #         estimated_labels[:,behs[b]] = rolled

        #     plt.plot(np.arange(0,len(probabilities)), threshold)
        #     plt.savefig(fn_pic)

        #probabilities = probabilities.clip(min=0, max=1.0)

        #estimated_labels = self.postprocessor(probabilities)

        # path, filename = os.path.split(outputfile)
        # fn = filename.split('_outputs')[0]
        # vid = os.path.join(path, fn+'.mp4')

        # cap = cv2.VideoCapture(vid)

        
        # # Check if the video opened successfully
        # if not cap.isOpened():
        #     print("Error: Couldn't open the video file.")
        #     return
        
        # ret, prev_frame = cap.read()
        # if not ret:
        #     print("Error: Couldn't read a frame from video.")
        #     return

        # # Convert the frame to grayscale
        # prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # motion_values = []
        # positions = []

        # while True:
        #     ret, frame = cap.read()
        #     if not ret:
        #         break

        #     # Convert the frame to grayscale
        #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
        #     # Compute the absolute difference between current frame and the previous frame
        #     diff = cv2.absdiff(prev_frame, gray_frame)

        #     diff = np.abs(gray_frame-prev_frame)

        #     num = diff.max()
        #     std = np.std(diff)

        #     diff[diff<50] = 0

        #     coords = np.column_stack(np.where(diff>num/2))
        #     centroid = np.mean(coords, axis=0)
        #     distances = np.linalg.norm(coords - centroid, axis=1)
        #     median_distance = np.mean(distances)

        #     if median_distance>50:
        #         diff[diff>num/2] = 0

        #     diff = diff - np.mean(diff)
        #     diff = np.abs(diff)

        #     # _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        #     # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #     # if contours:
        #     #     # Assuming the largest contour corresponds to the object of interest
        #     #     largest_contour = max(contours, key=cv2.contourArea)
        #     #     M = cv2.moments(largest_contour)

        #     #     if M["m00"] != 0:
        #     #         cX = int(M["m10"] / M["m00"])
        #     #         cY = int(M["m01"] / M["m00"])
        #     #         positions.append((cX, cY))
        #     #     else:
        #     #         positions.append((-1,-1))
            
        #     # else:
        #     #     positions.append((-1,-1))
        #     # Sum the differences to get a value representing the motion magnitudemotion_magnitude = np.mean(diff[diff>np.mean(diff)])
        #     motion_magnitude = np.mean(diff[diff>np.mean(diff)])
        #     if np.isnan(motion_magnitude):
        #         motion_magnitude = 0
        #     motion_values.append(motion_magnitude)

        #     # Update the previous frame
        #     prev_frame = gray_frame

        # motion_values = self.moving_average(np.array(motion_values))
        # cap.release()
        # motion_values = motion_values - motion_values.min()
        
        self.probabilities = probabilities
        self.estimated_labels = estimated_labels
        self.thresholds = thresholds


        if np.any(probabilities > 1) or np.any(estimated_labels > 1):
            print('Probabilities > 1 found, clamping...')
            probabilities = probabilities.clip(min=0, max=1.0)
            estimated_labels = estimated_labels.clip(min=0, max=1.0)
        
        """
            custom postprocessor based on amount of motion in a given frame
        """
        path, filename = os.path.split(outputfile)
        
        filename = filename.split('_outputs')[0]
        
        array = self.estimated_labels

        # for fn in range(0, len(motion_values)):
        #     print(fn)
        #     if motion_values[fn] < 4:
        #         print("changing to resting")

        #         for i in range(0,9):
        #             array[fn+49,i] = 0
        #             probabilities[fn+49,i]=thresholds[i]/2
        #         array[fn+49,0] = 0
        #         array[fn+49,7]=1

        #         probabilities[fn+49,7]=thresholds[7]
        #     if array[fn+49,0]==1 and motion_values[fn] < 30:
        #         print("doing the thing")
        #         array[fn+49,0] = 0
        #         array[fn+49,7] = 1
        #         probabilities[fn+49,7]=thresholds[7]
        
        # thresh = np.ones(len(array[:,5]))*thresholds[5] - probabilities[:,5]*.3
        # for fn in range(0,len(array[:,5])):
        #     if probabilities[fn,5]>thresh[fn] and array[fn,5]!=1:
        #         array[fn,5] = 1
        behs = {'background':0,'eating':1, 'drinking':2, 'rearing':3, 'climbing':4, 'digging':5, 'nesting':6, 'resting':7, 'grooming':8}
        behaviors = ['background','eating', 'drinking', 'rearing', 'climbing', 'digging', 'nesting', 'resting', 'grooming']


        # for b in behaviors:
        #     #fpr, tpr, thres = sklearn.metrics.roc_curve(df_labels[b].values, probabilities[:,behs[b]],pos_label=1)
        #     precision, recall, thres = sklearn.metrics.precision_recall_curve(df_labels[b].values[where_bg], probabilities[:,behs[b]][where_bg])

        #     fscore = 2*precision*recall/(precision+recall)
        #     # calculate the g-mean for each threshold
        #     #gmeans = np.sqrt(tpr * (1-fpr))
        #     # locate the index of the largest g-mean
        #     fscore[np.isnan(fscore)] = 0
        #     ix = np.argmax(fscore)
        #     print(b)
        #     print(fscore[ix])
        #     print(thres[ix])
        #     tholds[i] = thres[ix]
        #     #print(sklearn.metrics.auc(fpr,tpr))
        #     i+=1
        
        
        # for fn in range(0,len(array[:,0])):
        #     if array[fn,0] == 1:
        #         i = 0
        #         for b in behaviors:
        #             if b == "background":
        #                 i+=1
        #                 continue 
                    
        #             if b=="digging" and probabilities[fn,i] > tholds[i]:
        #                 array[fn,0] = 0
        #                 array[fn,i] = 1
                        
        #             i+=1
        # i = 0

        
        # tholds = np.copy(thresholds)
        # for b in behaviors:
        #     #fpr, tpr, thres = sklearn.metrics.roc_curve(df_labels[b].values, probabilities[:,behs[b]],pos_label=1)
        #     precision, recall, thres = sklearn.metrics.precision_recall_curve(df_labels[b].values, array[:,behs[b]])

        #     fscore = 2*precision*recall/(precision+recall)
        #     # calculate the g-mean for each threshold
        #     #gmeans = np.sqrt(tpr * (1-fpr))
        #     # locate the index of the largest g-mean
        #     fscore[np.isnan(fscore)] = 0
        #     ix = np.argmax(fscore)
        #     print(b)
        #     print(fscore[ix])
        #     print(thres[ix])
        #     tholds[i] = thres[ix]
        #     #print(sklearn.metrics.auc(fpr,tpr))
        #     i+=1
        
    
                
        self.estimated_labels = array
        
        self.latent_name = latent_name
        
    def make_internal_ethogram(self, n, t):
        array = self.estimated_labels
        
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)
        
        # translate this into graphing after infering all videos
        # print(self.videofile)
        # print(df.sum(axis=0))
        
        # should always be less than or equal to the video length
        resolution = 5
        frames = 3000
        vid_len = 5
        
        resolution_len = frames/vid_len * resolution
        
        times = frames/resolution_len 
        for i in range(0,int(times)):
            df_temp = df.loc[int(i*resolution_len+1):int((i+1)*resolution_len+1),:]
            self.diction['cam '+n].append([df_temp.sum(axis=0).append(pd.Series([t+i],index=['time']))])
    
    def make_bounded_ethogram(self, vid, n, t, bound_x, bound_y, bound_radius):
        array = self.estimated_labels
        
        
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)
        
        path, filename = os.path.split(os.path.splitext(vid)[0])
        filename = os.path.join(path, filename+'_motion.csv')
        
        df1 = pd.read_csv(filename)
        
        df = df.iloc[20:,:]
        
        total = pd.concat((df, df1), axis=1)
        
        
        # translate this into graphing after infering all videos
        # print(self.videofile)
        # print(df.sum(axis=0))
        
        # should always be less than or equal to the video length
        resolution = 5
        frames = 3000
        vid_len = 5
        
        resolution_len = frames/vid_len * resolution
        
        times = frames/resolution_len 
        for i in range(0,int(times)):
            df_temp = total.loc[int(i*resolution_len+1):int((i+1)*resolution_len+1),:]
            in_xbound = np.abs((128-df_temp['x_loc'].values) - bound_x)
            in_xbound = in_xbound < bound_radius
            in_bound_x = df_temp.loc[in_xbound]
            in_ybound = np.abs(in_bound_x['y_loc'].values - bound_y)
            in_ybound = in_ybound < bound_radius
            in_bound_xy = in_bound_x.loc[in_ybound]
            in_bound_xy = in_bound_xy.iloc[:,:9]
            self.diction['cam '+n].append([in_bound_xy.sum(axis=0).append(pd.Series([t+i],index=['time']))])
    
    
    def make_bounded_ethogram_exclude(self, vid, n, t, bound_x, bound_y, bound_radius):
        array = self.estimated_labels
        
        
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)
        
        path, filename = os.path.split(os.path.splitext(vid)[0])
        filename = os.path.join(path, filename+'_motion.csv')
        
        df1 = pd.read_csv(filename)
        
        df = df.iloc[20:,:]
        
        total = pd.concat((df, df1), axis=1)
        
        
        # translate this into graphing after infering all videos
        # print(self.videofile)
        # print(df.sum(axis=0))
        
        # should always be less than or equal to the video length
        resolution = 5
        frames = 3000
        vid_len = 5
        
        resolution_len = frames/vid_len * resolution
        
        times = frames/resolution_len 
        for i in range(0,int(times)):
            df_temp = total.loc[int(i*resolution_len+1):int((i+1)*resolution_len+1),:]
            neg = df_temp['x_loc'].values < 0
            df_temp = df_temp.loc[~neg]
            in_xbound = np.abs((128-df_temp['x_loc'].values) - bound_x)
            in_ybound = np.abs((df_temp['y_loc'].values) - bound_y)
            in_xbound = in_xbound < bound_radius
            in_ybound = in_ybound < bound_radius
            both = np.logical_or(~in_xbound,~in_ybound)
            
            in_bound_xy = df_temp.loc[both]
            in_bound_xy = in_bound_xy.iloc[:,:9]
            self.diction['cam '+n].append([in_bound_xy.sum(axis=0).append(pd.Series([t+i],index=['time']))])
        
    
    def export_predictions(self, n, t):

        array = self.estimated_labels
        
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)
        
        # translate this into graphing after infering all videos
        # print(self.videofile)
        # print(df.sum(axis=0))
        
        # should always be less than or equal to the video length
        """
        resolution = 5
        frames = 3000
        vid_len = 5
        resolution_len = frames/vid_len * resolution
        
        times = frames/resolution_len 
        for i in range(0,int(times)):
            df_temp = df.loc[int(i*resolution_len+1):int((i+1)*resolution_len+1),:]
            self.diction['cam '+n].append([df_temp.sum(axis=0).append(pd.Series([t+i],index=['time']))])
        """
        fname, _ = os.path.splitext(self.videofile)
        

        prediction_fname = fname+ '_predictions.csv'

        df.to_csv(prediction_fname)
        
    

    def get_selected_models(self, model_type: str = None):
        flow_model = None
        fe_model = None
        seq_model = None

        models = {'flow_generator': flow_model, 'feature_extractor': fe_model, 'sequence': seq_model}

        if not hasattr(self, 'trained_model_dict'):
            if model_type is not None:
                print('No weights found.')

            return models
            
        flow_text = '230321_210538_flow_generator_train'
        models['flow_generator'] = self.trained_model_dict['flow_generator'][flow_text]

        fe_text = '230612_224036_feature_extractor_train'
        models['feature_extractor'] = self.trained_model_dict['feature_extractor'][fe_text]

        seq_text = '230602_094633_sequence_train'
        models['sequence'] = self.trained_model_dict['sequence'][seq_text]

        return models
    
    def has_outputfile(records: dict) -> list:
        """ Convenience function for finding output files in a dictionary of records"""
        keys, has_outputs = [], []
        # check to see which records have outputfiles
        for key, record in records.items():
            keys.append(key)
            has_outputs.append(record['output'] is not None)
        return has_outputs
        
    def flow_inference(self):
        args = [
            'python', os.path.join('c:\\Users\\Jones-Lab\\Documents\\DEG','flow.py'), self.cfg.project.path
        ]
        self.inference_pipe = subprocess.Popen(args)
        self.inference_pipe.wait()
        
    def generate_sequence_inference_args(self):
        records = projects.get_records_from_datadir(self.data_path)
        keys = list(records.keys())
        outputs = projects.has_outputfile(records)
        sequence_weights = self.get_selected_models()['sequence']
        if sequence_weights is not None and os.path.isfile(sequence_weights):
            run_files = utils.get_run_files_from_weights(sequence_weights)
            sequence_config = OmegaConf.load(run_files['config_file'])
            # sequence_config = utils.load_yaml(os.path.join(os.path.dirname(sequence_weights), 'config.yaml'))
            latent_name = sequence_config['sequence']['latent_name']
            if latent_name is None:
                latent_name = sequence_config['feature_extractor']['arch']
            output_name = sequence_config['sequence']['output_name']
            if output_name is None:
                output_name = sequence_config['sequence']['arch']
        else:
            raise ValueError('must specify a valid weight file to run sequence inference!')

        # sequence_name, _ = utils.get_latest_model_and_name(self.project_config['project']['path'], 'sequence')

        # GOAL: MAKE ONLY FILES WITH LATENT_NAME PRESENT APPEAR ON LIST
        # SHOULD BE UNCHECKED IF THERE IS ALREADY THE "OUTPUT NAME" IN FILE

        has_latents = projects.do_outputfiles_have_predictions(self.data_path, latent_name)
        has_outputs = projects.do_outputfiles_have_predictions(self.data_path, output_name)
        no_sequence_outputs = [outputs[i] and not has_outputs[i] for i in range(len(records))]
        keys_with_features = []
        for i, key in enumerate(keys):
            if has_latents[i]:
                keys_with_features.append(key)
        
        should_infer = no_sequence_outputs
        all_false = np.all(np.array(should_infer) == False)
        if all_false:
            return
        weights = self.get_selected_models()['sequence']
        if weights is not None and os.path.isfile(weights):
            weight_arg = 'sequence.weights={}'.format(weights)
        else:
            raise ValueError('weights do not exist! {}'.format(weights))
        args = [
            'python', '-m', 'deepethogram.sequence.inference', 'project.path={}'.format(self.cfg.project.path),
            'inference.overwrite=True', weight_arg
        ]
        string = 'inference.directory_list=['
        for key, infer in zip(keys, should_infer):
            if infer:
                record_dir = os.path.join(self.data_path, key) + ','
                string += record_dir
        string = string[:-1] + ']'
        args += [string]
        return args

    def sequence_infer(self):

        args = self.generate_sequence_inference_args()
        if args is None:
            return
            
        self.inference_pipe = subprocess.Popen(args)
        self.inference_pipe.wait()
        
        if self.inference_pipe.poll() is None:
            self.inference_pipe.terminate()
            self.inference_pipe.wait()
            
        del self.inference_pipe

    def generate_featureextractor_inference_args(self):
        records = projects.get_records_from_datadir(self.data_path)
        keys, no_outputs = [], []
        for key, record in records.items():
            keys.append(key)
            no_outputs.append(record['output'] is None)

        should_infer = no_outputs
        all_false = np.all(np.array(should_infer) == False)

        if all_false:
            return
            
        weights = self.get_selected_models()['feature_extractor']
        if weights is not None and os.path.isfile(weights):
            weight_arg = 'feature_extractor.weights={}'.format(weights)
        else:
            raise ValueError('Dont run inference without using a proper feature extractor weights! {}'.format(weights))

        args = [
            'python', '-m', 'deepethogram.feature_extractor.inference', 'project.path={}'.format(self.cfg.project.path),
            'inference.overwrite=False', weight_arg
        ]
        flow_weights = self.get_selected_models()['flow_generator']
        assert flow_weights is not None
        args += ['flow_generator.weights={}'.format(flow_weights)]
        string = 'inference.directory_list=['
        for key, infer in zip(keys, should_infer):
            if infer:
                record_dir = os.path.join(self.data_path, key) + ','
                string += record_dir
        string = string[:-1] + ']'
        args += [string]
        return args
        
    def generate_fe_videolist(self):
        records = projects.get_records_from_datadir(self.data_path)
        keys, no_outputs = [], []
        for key, record in records.items():
            keys.append(key)
            no_outputs.append(record['output'] is None)
        
        records = []
        for key, infer in zip(keys, no_outputs):
            if infer:
                record_dir = os.path.join(self.data_path, key)
                records.append(os.path.join(record_dir, key+'.mp4'))

        return records
        

    def featureextractor_infer(self):
        args = self.generate_featureextractor_inference_args()
            
        self.inference_pipe = subprocess.Popen(args)
        self.inference_pipe.wait()
    
    def load_predictions_wl(self, white_list):
    
        dp = self.data_path
        filenames = []
        for root, dirs, files in os.walk(dp):
            for dir in dirs:
                video_name = os.path.join(os.path.join(dp,dir),dir+'.mp4')
                if os.path.isfile(video_name) and dir in white_list:
                
                    created_time = os.path.getctime(video_name)
                    filenames.append((video_name, created_time))
        for filename in filenames:
        
            ts = filename[1]
            
            _,filename = os.path.split(os.path.splitext(filename[0])[0])
            
            # actual post processing
            self.import_outputfile(os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5'))
            
            self.videofile = os.path.join(os.path.join(self.data_path,filename),filename+'.mp4')
            num = filename.split('_')[1]
            
            dt = datetime.fromtimestamp(ts).time()
            t = dt.hour*60 + dt.minute
            
            self.export_predictions(num,t)
        
    def load_predictions(self):
    
        dp = self.data_path
        filenames = []
        for root, dirs, files in os.walk(dp):
            for dir in dirs:
                video_name = os.path.join(os.path.join(dp,dir),dir+'.mp4')
                output_name = os.path.join(os.path.join(dp,dir),dir+'_outputs.h5')
                #motion_name = os.path.join(os.path.join(dp,dir),dir+'_motion.csv')
                pred_name = os.path.join(os.path.join(dp,dir),dir+'_predictions.csv')
                if os.path.isfile(video_name) and os.path.isfile(output_name):
                    if os.path.isfile(pred_name):
                        continue
                    created_time = os.path.getctime(video_name)
                    filenames.append((video_name, created_time))
                    
        for filename in filenames:
        
            ts = filename[1]
            #print(filename[0])
            
            _,filename = os.path.split(os.path.splitext(filename[0])[0])
            outputs_path = os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5')
            pred_path = os.path.join(os.path.join(self.data_path,filename),filename+'_predictions.csv')
            
            if os.path.isfile(outputs_path) and not os.path.isfile(pred_path):
            
                #print(os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5'))
                self.import_outputfile(os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5'))
                self.videofile = os.path.join(os.path.join(self.data_path,filename),filename+'.mp4')
                num = filename.split('_')[1]
                
                dt = datetime.fromtimestamp(ts).time()
                t = dt.hour*60 + dt.minute
                
                self.export_predictions(num,t)
        
        #print(self.diction)
    def load_bounded_predictions(self, x, y, r):
    
        dp = self.data_path
        filenames = []
        for root, dirs, files in os.walk(dp):
            for dir in dirs:
                video_name = os.path.join(os.path.join(dp,dir),dir+'.mp4')
                if os.path.isfile(video_name):
                
                    created_time = os.path.getctime(video_name)
                    filenames.append((video_name, created_time))
        for filename in filenames:
        
            ts = filename[1]
            #print(filename[0])
            
            _,filename = os.path.split(os.path.splitext(filename[0])[0])
            #print(os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5'))
            self.import_outputfile(os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5'))
            self.videofile = os.path.join(os.path.join(self.data_path,filename),filename+'.mp4')
            num = filename.split('_')[1]
            
            dt = datetime.fromtimestamp(ts).time()
            t = dt.hour*60 + dt.minute
            
            self.make_bounded_ethogram_exclude(self.videofile, '1',t, x, y, r)
        
        #print(self.diction)
        
            
    def reset_infer(self):
        self.listener = None 
        self.inference_pipe = None
        
        
    def feature_infer(self):
        args = self.generate_featureextractor_inference_args()
        
        if args==None:
            return
        
        rgb = self.generate_fe_videolist()
        
        if len(rgb)==0:
            return
        
        if not self.built_fe:
            project_path = projects.get_project_path_from_cl(args[3:])
            self.fe_cfg = fei.make_feature_extractor_inference_cfg(project_path, use_command_line=True)
            self.fe = fei.feature_extractor_inference(self.fe_cfg)
            self.built_fe = True
        else:
            device = 'cuda:{}'.format(self.fe_cfg.compute.gpu_id)
            fei.extract(rgb,
                self.fe[0],
                final_activation=self.fe[1],
                thresholds=self.fe[2],
                postprocessor=self.fe[3],
                mean_by_channels=self.fe[4],
                fusion=self.fe[5],
                num_rgb=self.fe[6],
                latent_name=self.fe[7],
                device=device,
                cpu_transform=self.fe[9],
                gpu_transform=self.fe[10],
                ignore_error=self.fe[11],
                overwrite=self.fe[12],
                class_names=self.fe[13],
                num_workers=self.fe[14],
                batch_size=self.fe[15])

    

class JobListener(threading.Thread):
    def __init__(self, pipe, completed_jobs):
        threading.Thread.__init__(self)
        self.pipe = pipe
        self.completed_jobs = completed_jobs
        self.should_continue = True

    def run(self):
        while self.should_continue:
            time.sleep(1)
            if self.pipe.poll() is not None:
                # The job is complete, add it to the queue
                print("finished with the jobs!")
                
                break
                
    def wait(self):
        time.sleep(2)

    def stop(self):
        self.should_continue = False
        
def plot_single_predictions(fig, ax, hd, n):
    # order is as follows: resting, eating, drinking, climbing, rearing, digging, grooming, nesting
    # need a function to plot the actogram of stacked bars, one for each behavior
    w = 7
    fcount = 3000
    for cam in list(hd.diction.keys()):
        num = int(cam.split('cam ')[1])-1
        if n != num+1:
            continue
        ax.cla()
        ax.set_xlim([0,1440])
        ax.set_ylim([0,50])
        ax.set_ylabel('cam '+str(num+1))
        ax.axvspan(0,410,facecolor='darkgray',alpha=1)
        ax.axvspan(410,1130,facecolor='yellow',alpha=0.2)
        ax.axvspan(1130,1440,facecolor='darkgray',alpha=1)
        ax.axhspan(0,45,facecolor='white',alpha=1)
        
        ax.set_yticks([i for i in np.arange(5,50,5)],['background','eating','drinking','resting','nesting','grooming','digging','rearing','climbing'])
        ax.grid(color='black',axis='y',linestyle='-',linewidth=0.5,markevery=(5,5,50))
        plt.setp(ax.yaxis.get_majorticklabels(),rotation=30)
        
        if len(hd.diction[cam]) is not 0:
                
            background = []
            eating = []
            drinking = []
            resting = []
            nesting = []
            grooming = []
            digging = []
            rearing = []
            climbing = []
            times = []
            
            for l in hd.diction[cam]:
                background.append(l[0]['background'])
                eating.append(l[0]['eating'])
                drinking.append(l[0]['drinking'])
                resting.append(l[0]['resting'])
                nesting.append(l[0]['nesting'])
                grooming.append(l[0]['grooming'])
                digging.append(l[0]['digging'])
                rearing.append(l[0]['rearing'])
                climbing.append(l[0]['climbing'])
                times.append(l[0]['time']%1440)
            
            bga = (np.array(background)/fcount)
            bga = bga+.2
            bga[bga>1] = 1
            height = [5 for i in range(0,len(background))]
            rgba_bg = np.zeros((len(background),4))
            heat = np.array(background)/(np.amax(np.array(background))+1)
            heat[heat>1] = 1
            rgba_bg[:,0] = bga
            rgba_bg[:,1] = 0
            rgba_bg[:,2] = 0
            rgba_bg[:,3] = heat
                
            ax.bar(times,height,width=w,label='background', color=rgba_bg)
            bot = [height[i] for i in range(0,len(height))]
            
            alps = (np.array(eating)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(eating))]
            rgba = np.zeros((len(eating),4))
            heat = np.array(eating)/(np.amax(np.array(eating))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='eating', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(drinking)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(drinking))]
            rgba = np.zeros((len(drinking),4))
            heat = np.array(drinking)/(np.amax(np.array(drinking))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='drinking', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(resting)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(resting))]
            rgba = np.zeros((len(resting),4))
            heat = np.array(resting)/(np.amax(np.array(resting))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='resting', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(nesting)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(nesting))]
            rgba = np.zeros((len(nesting),4))
            heat = np.array(nesting)/(np.amax(np.array(nesting))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='nesting', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(grooming)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(grooming))]
            rgba = np.zeros((len(grooming),4))
            heat = np.array(grooming)/(np.amax(np.array(grooming))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='grooming', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(digging)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(digging))]
            rgba = np.zeros((len(digging),4))
            heat = np.array(digging)/(np.amax(np.array(digging))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='digging',color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(rearing)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(rearing))]
            rgba = np.zeros((len(rearing),4))
            heat = np.array(rearing)/(np.amax(np.array(rearing))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='rearing',color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(climbing)/(fcount - np.array(background)+1))
            alps = alps+.2
            alps[alps>1] = 1
            height = [5 for i in range(0,len(climbing))]
            rgba = np.zeros((len(climbing),4))
            heat = np.array(climbing)/(np.amax(np.array(climbing))+1)
            heat[heat>1] = 1
            rgba[:,0] = alps
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = heat
            
            ax.bar(times,height,bottom=bot,width=w,label='climbing',color=rgba)
        
def plot_positional_behaviors(fig, ax, hd, filenames):
    
    # order is as follows: resting, eating, drinking, climbing, rearing, digging, grooming, nesting
    # need a function to plot the actogram of stacked bars, one for each behavior
    
    filenames = sorted(filenames, key=lambda f:f[1])
    
    r = 1
    behs = {'background':'white','eating':'red','drinking':'green','rearing':'orange','climbing':'yellow','digging':'purple','nesting':'lime','resting':'blue','grooming':'cyan'}
    behaves = ['background','eating','drinking','rearing','climbing','digging','nesting','resting','grooming']
    
    for b in behaves:
        ax.cla()
        ax.set_xlim([0,128])
        ax.set_ylim([0,128])
        ax.set_aspect('equal')
        ax.set_facecolor('black')
        ax.set_title(b)
        for filename in filenames:
        
            _,filename = os.path.split(os.path.splitext(filename[0])[0])
            motions = os.path.join(os.path.join(hd.data_path,filename),filename+'_motion.csv')
            predictions = os.path.join(os.path.join(hd.data_path,filename),filename+'_predictions.csv')
            
            mts = pd.read_csv(motions)
            pred = pd.read_csv(predictions).loc[20:,:]
            
            
            total = pd.concat((pred, mts), axis=1)
            
            l = len(total.loc[total[b]==1, 'x_loc'].values)
            ax.scatter(x=(128-total.loc[total[b]==1, 'x_loc'].values),y=total.loc[total[b]==1, 'y_loc'].values, marker='o', s=(np.ones(l)+24),color=behs[b], alpha=.1)
        plt.pause(10)
            
        
        
def plot_predictions(fig, axs, hd):
        
    # order is as follows: resting, eating, drinking, climbing, rearing, digging, grooming, nesting
    # need a function to plot the actogram of stacked bars, one for each behavior
    numdays = 4
    w = 30
    for cam in list(hd.diction.keys()):
        num = int(cam.split('cam ')[1])-1
        axs[int(num/4),num%4].cla()
        axs[int(num/4),num%4].set_xlim([0,1440])
        axs[int(num/4),num%4].set_ylim([0,50])
        axs[int(num/4),num%4].set_ylabel('cam '+str(num+1))
        axs[int(num/4),num%4].axvspan(0,410,facecolor='darkgray',alpha=1)
        axs[int(num/4),num%4].axvspan(410,1130,facecolor='yellow',alpha=0.2)
        axs[int(num/4),num%4].axvspan(1130,1440,facecolor='darkgray',alpha=1)
        axs[int(num/4),num%4].axhspan(0,45,facecolor='white',alpha=1)
        
        axs[int(num/4),num%4].set_yticks([i for i in np.arange(5,50,5)],['background','eating','drinking','resting','nesting','grooming','digging','rearing','climbing'])
        axs[int(num/4),num%4].grid(color='black',axis='y',linestyle='-',linewidth=0.5,markevery=(5,5,50))
        plt.setp(axs[int(num/4),num%4].yaxis.get_majorticklabels(),rotation=30)
        
        if len(hd.diction[cam]) is not 0:
                
            background = []
            eating = []
            drinking = []
            resting = []
            nesting = []
            grooming = []
            digging = []
            rearing = []
            climbing = []
            times = []
            
            for l in hd.diction[cam]:
                background.append(l[0]['background'])
                eating.append(l[0]['eating'])
                drinking.append(l[0]['drinking'])
                resting.append(l[0]['resting'])
                nesting.append(l[0]['nesting'])
                grooming.append(l[0]['grooming'])
                digging.append(l[0]['digging'])
                rearing.append(l[0]['rearing'])
                climbing.append(l[0]['climbing'])
                times.append(l[0]['time']%1440)
            
            bga = (np.array(background)/18000)
            bga = bga+.1
            bga[bga>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(background))]
            rgba_bg = np.zeros((len(background),4))
            heat = np.array(background)/(np.amax(np.array(background))+1)
            heat[heat>1] = 1
            rgba_bg[:,0] = heat
            rgba_bg[:,1] = 0
            rgba_bg[:,2] = 0
            rgba_bg[:,3] = bga
                
            axs[int(num/4),num%4].bar(times,height,width=w,label='background', color=rgba_bg)
            bot = [height[i] for i in range(0,len(height))]
            
            alps = (np.array(eating)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(eating))]
            rgba = np.zeros((len(eating),4))
            heat = np.array(eating)/(np.amax(np.array(eating))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='eating', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(drinking)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(drinking))]
            rgba = np.zeros((len(drinking),4))
            heat = np.array(drinking)/(np.amax(np.array(drinking))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='drinking', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(resting)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(resting))]
            rgba = np.zeros((len(resting),4))
            heat = np.array(resting)/(np.amax(np.array(resting))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='resting', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(nesting)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(nesting))]
            rgba = np.zeros((len(nesting),4))
            heat = np.array(nesting)/(np.amax(np.array(nesting))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='nesting', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(grooming)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(grooming))]
            rgba = np.zeros((len(grooming),4))
            heat = np.array(grooming)/(np.amax(np.array(grooming))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='grooming', color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(digging)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(digging))]
            rgba = np.zeros((len(digging),4))
            heat = np.array(digging)/(np.amax(np.array(digging))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='digging',color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(rearing)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(rearing))]
            rgba = np.zeros((len(rearing),4))
            heat = np.array(rearing)/(np.amax(np.array(rearing))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='rearing',color=rgba)
            bot = [bot[i]+height[i] for i in range(0,len(bot))]
            
            alps = (np.array(climbing)/(18000 - np.array(background)+1))
            alps = alps+.1
            alps[alps>1/numdays] = 1/numdays
            height = [5 for i in range(0,len(climbing))]
            rgba = np.zeros((len(climbing),4))
            heat = np.array(climbing)/(np.amax(np.array(climbing))+1)
            heat[heat>1] = 1
            rgba[:,0] = heat
            rgba[:,1] = 0
            rgba[:,2] = 0
            rgba[:,3] = alps
            
            axs[int(num/4),num%4].bar(times,height,bottom=bot,width=w,label='climbing',color=rgba)
        

if __name__ == '__main__':
    hd = headless_deg(sys.argv[1])

    black_list = []

    baseDirectory = 'C:\\Users\\Jones-Lab\\Documents\\Python Scripts\\videos'
    
    fig, axs = plt.subplots(4,4,figsize=(17,10))
    
    fig_name = os.path.join(os.path.split(sys.argv[1])[0], 'ethogram.png')
    
    first = True
    
    num_cams = 1

    while True:
        videos_to_add = []
        hd.load_predictions()
        
        print("Finding videos that are ready...")

        n = 1
        while n<=16:

            # Specify the directory to check for files
            directory = baseDirectory + '/cam'+str(n)+'/*'

            # Get the current time
            now = time.time()

            # Get a list of all files in the directory
            files = glob.glob(directory)

            # Check each file
            for file in files:
                # Get the time of the last modification of the file
                last_modified_time = os.path.getmtime(file)
                created_time = os.path.getctime(file)
                print(file)

                ext = os.path.splitext(file)[1]
                if ext!='.mp4':
                    continue

                # If the file was last modified more than 30 seconds ago, try to infer it
                print(now - created_time)
                #now - created_time > 300 and
                if not file in black_list:
                    print(now - created_time)
                    videos_to_add.append((file, created_time))
                
            
            n+=1
        
        if len(videos_to_add) >= num_cams:
            for file in videos_to_add:
                black_list.append(file[0])
        else:
            time.sleep(20)
            continue
            
            
        print("Found {} videos!".format(len(videos_to_add)))
        if len(videos_to_add) is not 0:
            try:
                hd.add_multiple_videos(list(videos_to_add))
            except:
                continue
        
        now1 = time.time()
        if len(videos_to_add) is not 0 or first:
            first = False
            try:
                hd.feature_infer()
                #hd.sequence_infer()
                #hd.flow_inference()
            except:
                print("error in inferencing")
                continue
        
        now2 = time.time()
        
        
        # this function will store the sums of labels across the five minute video
        try:
            wl = []
            for v in videos_to_add:
                wl.append(os.path.splitext(os.path.split(v[0])[1])[0])
            hd.load_predictions_wl(wl)  
        except:
            print("error in loading predictions")
            continue
        
        now3 = time.time()    

        print("video load time " + str(now1-now))
        print("video inference time " + str(now2-now1))
        print("video post process time " + str(now3-now2))
        print("video total time " + str(now3-now))
    
            
        time.sleep(20)

    
        
        
    
    
