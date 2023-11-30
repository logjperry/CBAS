
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


        self.postprocessor = get_postprocessor_from_cfg(self.cfg, thresholds)


     
        self.probabilities = probabilities
        self.estimated_labels = estimated_labels
        self.thresholds = thresholds


        if np.any(probabilities > 1) or np.any(estimated_labels > 1):
            print('Probabilities > 1 found, clamping...')
            probabilities = probabilities.clip(min=0, max=1.0)
            estimated_labels = estimated_labels.clip(min=0, max=1.0)
        
        path, filename = os.path.split(outputfile)
        
        filename = filename.split('_outputs')[0]
        
        self.latent_name = latent_name
        
    
    def export_predictions(self, n, t):

        array = self.estimated_labels
        
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)
        
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

    
        
        
    
    
