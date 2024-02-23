
import os
import glob
import time
import traceback
import subprocess
import threading
import numpy as np
import pandas as pd
from queue import Queue
from typing import Union
import sys
from omegaconf import DictConfig, OmegaConf
import math
import json
from datetime import datetime
import yaml




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

        self.noseq = False

        self.initialize_project(project_head)

    def initialize_project(self, directory: Union[str, os.PathLike]):
        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        if len(directory) == 0:
            return
        filename = os.path.join(directory, 'project_config.yaml')

        if len(filename) == 0 or not os.path.isfile(filename):
            raise Exception('Could not load the project config file at: {}'.format(filename))


        # overwrite cfg passed at command line now that we know the project path. still includes command line arguments
        self.cfg = deepethogram.configuration.make_config(directory, ['config', 'gui', 'postprocessor'], run_type='gui', model=None)


        self.cfg = projects.convert_config_paths_to_absolute(self.cfg, raise_error_if_pretrained_missing=False)


        self.cfg = projects.setup_run(self.cfg, raise_error_if_pretrained_missing=False)



        # for convenience
        self.data_path = self.cfg.project.data_path
        self.model_path = self.cfg.project.model_path

        self.get_trained_models()

        self.get_selected_models()

    def get_trained_models(self):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

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

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei
        
        if 'preset' in self.cfg:
            preset = self.cfg.preset
        else:
            preset = 'deg_f'
        default_archs = projects.load_default('preset/{}'.format(preset))
        seq_default = projects.load_default('model/sequence')
        default_archs['sequence'] = {'arch': seq_default['sequence']['arch']}

        if 'feature_extractor' in self.cfg and self.cfg.feature_extractor.arch != None:
            default_archs['feature_extractor']['arch'] = self.cfg.feature_extractor.arch
        if 'flow_generator' in self.cfg and self.cfg.flow_generator.arch != None:
            default_archs['flow_generator']['arch'] = self.cfg.flow_generator.arch
        if 'sequence' in self.cfg and 'arch' in self.cfg.sequence and self.cfg.sequence.arch != None:
            default_archs['sequence']['arch'] = self.cfg.sequence.arch
        self.default_archs = default_archs

    def add_multiple_videos(self, filenames):
        if self.data_path != None:
            data_dir = self.data_path
        else:
            raise ValueError('create or load a DEG project before loading video')
        if len(filenames) == 0:
            return
        for filename in filenames:
            assert os.path.exists(filename)

        for filename in filenames:
            self.initialize_video(filename)

    def initialize_video(self, videofile: Union[str, os.PathLike]):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei
        
        try: 

            if  os.path.normpath(self.cfg.project.data_path) in os.path.normpath(videofile) and projects.is_deg_file(videofile):

                record = projects.get_record_from_subdir(os.path.dirname(videofile))
                
                labelfile = record['label']
                outputfile = record['output']

                self.labelfile = labelfile
                self.outputfile = outputfile

                if outputfile != None:
                    self.import_outputfile(outputfile, first_time=True)
            else:
            
                new_loc = projects.add_video_to_project(OmegaConf.to_container(self.cfg), videofile, mode='copy')
                
                self.videofile = new_loc
                
                utils.load_yaml(os.path.join(os.path.dirname(self.videofile), 'record.yaml'))



        except BaseException as e:
            tb = traceback.format_exc()
            print(tb)
            return
        
    def get_selected_models(self, model_type: str = None):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        flow_model = None
        fe_model = None
        seq_model = None

        models = {'flow_generator': flow_model, 'feature_extractor': fe_model, 'sequence': seq_model}
        

        if not hasattr(self, 'trained_model_dict'):
            if model_type != None:
                print('No weights found.')

            return models

            
        
        try:
            flow_text = self.cfg['flow_generator']['cbas']
            models['flow_generator'] = self.trained_model_dict['flow_generator'][flow_text]

            fe_text = self.cfg['feature_extractor']['cbas']
            models['feature_extractor'] = self.trained_model_dict['feature_extractor'][fe_text]

            seq_text = self.cfg['sequence']['cbas']

            if seq_text == None:
                seq_text = 'latest'
                self.noseq = True

            models['sequence'] = self.trained_model_dict['sequence'][seq_text]

            print(f'Loading flow_generator={flow_text}, feature_extractor={fe_text}, sequence_model={seq_text}')
        except:
            raise Exception(f'Could not find a valid models matching the names initially provided. Remove this model and reload it to select the models to be used.')

        return models
    
    def has_outputfile(records: dict) -> list:

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        """ Convenience function for finding output files in a dictionary of records"""
        keys, has_outputs = [], []
        # check to see which records have outputfiles
        for key, record in records.items():
            keys.append(key)
            has_outputs.append(record['output'] != None)
        return has_outputs
        
    def generate_sequence_inference_args(self):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei


        records = projects.get_records_from_datadir(self.data_path)
        keys = list(records.keys())
        outputs = projects.has_outputfile(records)
        sequence_weights = self.get_selected_models()['sequence']
        if sequence_weights != None and os.path.isfile(sequence_weights):
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
        if weights != None and os.path.isfile(weights):
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

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        if self.noseq:
            return

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

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

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
        if weights != None and os.path.isfile(weights):
            weight_arg = 'feature_extractor.weights={}'.format(weights)
        else:
            raise ValueError('Dont run inference without using a proper feature extractor weights! {}'.format(weights))

        args = [
            'python', '-m', 'deepethogram.feature_extractor.inference', 'project.path={}'.format(self.cfg.project.path),
            'inference.overwrite=False', weight_arg
        ]
        flow_weights = self.get_selected_models()['flow_generator']
        assert flow_weights != None
        args += ['flow_generator.weights={}'.format(flow_weights)]
        string = 'inference.directory_list=['
        for key, infer in zip(keys, should_infer):
            if infer:
                record_dir = os.path.join(self.data_path, key) + ','
                string += record_dir
        string = string[:-1] + ']'
        args += [string]
        return args
           
    def feature_infer(self, video_list):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        args = self.generate_featureextractor_inference_args()
        
        if args==None:
            return
        
        rgb = [os.path.join(os.path.join(self.data_path,vid),vid+'.mp4') for vid in video_list]
        
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
            
    def load_predictions(self, video_list):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei
    
        dp = self.data_path
        filenames = []
        for root, dirs, files in os.walk(dp):
            for dir in dirs:
                video_name = os.path.join(os.path.join(dp,dir),dir+'.mp4')
                if os.path.isfile(video_name) and dir in video_list:
                
                    created_time = os.path.getctime(video_name)
                    filenames.append((video_name, created_time))

        for filename in filenames:
            
            _,filename = os.path.split(os.path.splitext(filename[0])[0])
            
            self.import_outputfile(os.path.join(os.path.join(self.data_path,filename),filename+'_outputs.h5'))
            
            self.videofile = os.path.join(os.path.join(self.data_path,filename),filename+'.mp4')
            
            self.export_predictions()
            
    def import_outputfile(self, outputfile: Union[str, os.PathLike], latent_name=None):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        if outputfile is None:
            return
        try:
            outputs = projects.import_outputfile(self.cfg.project.path,
                                                 outputfile,
                                                 class_names=OmegaConf.to_container(self.cfg.project.class_names),
                                                 latent_name=latent_name)
        except ValueError as e:
            print('If you got a broadcasting error: did you add or remove behaviors and not re-train?')
            return

        probabilities, thresholds, latent_name, keys = outputs


        self.postprocessor = get_postprocessor_from_cfg(self.cfg, thresholds)

        estimated_labels = self.postprocessor(probabilities)
     
        self.probabilities = probabilities
        self.estimated_labels = estimated_labels
        self.thresholds = thresholds


        if np.any(probabilities > 1) or np.any(estimated_labels > 1):
            print('Probabilities > 1 found, clamping...')
            probabilities = probabilities.clip(min=0, max=1.0)
            estimated_labels = estimated_labels.clip(min=0, max=1.0)
        
        
        self.latent_name = latent_name
        
    def export_predictions(self):

        import deepethogram
        from cbas_headless.platforms import modified_deepethogram
        from cbas_headless.platforms.modified_deepethogram import projects
        from deepethogram import utils
        from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
        from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

        array = self.estimated_labels       
        df = pd.DataFrame(data=array, columns=self.cfg.project.class_names)      
        fname, _ = os.path.splitext(self.videofile)    
        prediction_fname = fname+ '_predictions.csv'
        df.to_csv(prediction_fname)


def inference(hl_deg, name, video_path, postprocessor, postprocessor_path):

    import deepethogram
    from cbas_headless.platforms import modified_deepethogram
    from cbas_headless.platforms.modified_deepethogram import projects
    from deepethogram import utils
    from cbas_headless.platforms.modified_deepethogram.postprocessing import get_postprocessor_from_cfg
    from cbas_headless.platforms.modified_deepethogram.feature_extractor import inference as fei

    
    from postprocessor import lstm_classifier
    from postprocessor import frame_encoder

    
    sys.modules['frame_encoder'] = frame_encoder
    sys.modules['lstm_classifier'] = lstm_classifier

    while True:

        ready = []
        finished = []
        wrapup = False
        lock_file = None


        if os.path.exists(video_path):
            with open(video_path, 'r') as file:
                iconfig = yaml.safe_load(file)
                finished = iconfig['finished']
                ready = iconfig['ready']
                wrapup = iconfig['wrapup']
                lock_file = iconfig['lock']
        else:
            time.sleep(5)
            continue


        with open(lock_file, 'r') as file:
            lconfig = yaml.safe_load(file)
        lock = lconfig['lock']
        
        if lock != name:
            done = False
            while not done:
                time.sleep(5)
                with open(lock_file, 'r') as file:
                    lconfig = yaml.safe_load(file)
                lock = lconfig['lock']
        
                if lock == name:
                    done = True
                    
        # start the timer
        start_time = time.time()
        
        with open(video_path, 'r') as file:
            iconfig = yaml.safe_load(file)
            finished = iconfig['finished']
            ready = iconfig['ready']
            wrapup = iconfig['wrapup']
            lock_file = iconfig['lock']
            times = iconfig['times']
            avg_times = iconfig['avg_times']
                    
                            

        videos_to_add = ready

        if len(videos_to_add)==0 and wrapup:
            lconfig = {'lock':None,'dead':lconfig['dead']}
            with open(lock_file, 'w') as file:
                yaml.dump(lconfig, file, allow_unicode=True)
            break
        elif wrapup and lock != name:
            lconfig = {'lock':None,'dead':lconfig['dead']}
            with open(lock_file, 'w') as file:
                yaml.dump(lconfig, file, allow_unicode=True)
            break

        if len(videos_to_add) != 0:
            success = False

            # Try to add all of the videos at once
            try:
                hl_deg.add_multiple_videos(videos_to_add)
                success = True
            except:
                print('Error in loading the videos... Trying to isolate the problem video(s).')

            if not success:
                good_videos = []
                for vid in videos_to_add:
                    sub_list = [vid]
                    try:
                        hl_deg.add_multiple_videos(sub_list)
                        good_videos.append(vid)
                    except:
                        print(f'There is a problem with loading this video, {vid}.')
            
        
        # turn the videos into directory names for easier finding within deg structure
        videos_to_add = [os.path.split(os.path.splitext(vid)[0])[1] for vid in videos_to_add]
        
        if len(videos_to_add) != 0:
            success = False

            # Try to infer all of the videos at once
            try:
                hl_deg.feature_infer(videos_to_add)

                if postprocessor=='':
                    hl_deg.sequence_infer()
                else:
                    videos_to_add = [os.path.join(hl_deg.data_path, vid, vid+'.mp4') for vid in videos_to_add]

                    with open(postprocessor_path, 'r') as file:
                        ppconfig = yaml.safe_load(file)

                    models = ppconfig['models']


                    encoder_config = os.path.splitext(models[postprocessor]['encoder'])[0]+'.yaml'
                    classifier_config = os.path.splitext(models[postprocessor]['classifier'])[0]+'.yaml'

                    
                    vids = []
                    for v in videos_to_add:
                        if os.path.exists(os.path.splitext(v)[0]+"_outputs.h5"):
                            vids.append(v)
                    frame_encoder.infer(encoder_config, hl_deg.data_path, vids)


                    vids = []
                    for v in videos_to_add:
                        if os.path.exists(os.path.splitext(v)[0]+"_outputs_encoded_features.h5"):
                            vids.append(v)
                    lstm_classifier.infer(classifier_config, hl_deg.data_path, videos_to_add)

                    videos_to_add = [os.path.split(os.path.splitext(vid)[0])[1] for vid in videos_to_add]

                success = True
            except:
                print('Error in inferring the videos... Trying to isolate the problem video(s).')

            if not success:
                good_videos = []
                for vid in videos_to_add:
                    sub_list = [vid]
                    try:
                        hl_deg.feature_infer(sub_list)
                        if postprocessor=='':
                            hl_deg.sequence_infer()
                        else:
                            sub_list = [os.path.join(hl_deg.data_path, vid, vid+'.mp4') for vid in sub_list]

                            with open(postprocessor_path, 'r') as file:
                                ppconfig = yaml.safe_load(file)

                            models = ppconfig['models']

                            encoder_config = os.path.splitext(models[postprocessor]['encoder'])[0]+'.yaml'
                            classifier_config = os.path.splitext(models[postprocessor]['classifier'])[0]+'.yaml'

                            sub_list_verified = []
                            for v in sub_list:
                                if os.path.exists(os.path.splitext(v)[0]+"_outputs.h5"):
                                    sub_list_verified.append(v)

                            if len(sub_list_verified)!=0:
                                frame_encoder.infer(encoder_config, hl_deg.data_path, sub_list_verified)
                            sub_list = sub_list_verified.copy()

                            sub_list_verified = []
                            for v in sub_list:
                                if os.path.exists(os.path.splitext(v)[0]+"_outputs_encoded_features.h5"):
                                    sub_list_verified.append(v)

                            if len(sub_list_verified)!=0:
                                lstm_classifier.infer(classifier_config, hl_deg.data_path, sub_list_verified)

                        good_videos.append(vid)
                    except:
                        print(f'There is a problem with inferring this video, {vid}.')
            

        
        if len(videos_to_add) != 0:
            success = False

            # Try to load predictions for all of the videos at once
            try:
                hl_deg.load_predictions(videos_to_add)
                success = True
            except:
                print('Error in loading predictions for the videos... Trying to isolate the problem video(s).')

            if not success:
                good_videos = []
                for vid in videos_to_add:
                    sub_list = [vid]
                    try:
                        hl_deg.load_predictions(sub_list)
                        good_videos.append(vid)
                    except:
                        print(f'There is a problem with loading predictions for this video, {vid}.')


        for vid in videos_to_add:
            real_name = os.path.join(hl_deg.data_path, vid)
            if real_name not in finished:
                finished.append(real_name)
        
        # calculate the total time
        total_time = time.time() - start_time
        avg_time = 0
    
        if len(videos_to_add)>0:
            avg_time = total_time/len(videos_to_add)

        
        with open(video_path, 'r') as file:
            iconfig = yaml.safe_load(file)
            wrapup = iconfig['wrapup']

        
        times.append(total_time)
        avg_times.append(avg_time)

        iconfig = {'ready':[],'finished':finished,'wrapup':wrapup,'lock':lock_file, 'avg_times':avg_times, 'times':times, 'iter':iconfig['iter']+1}

            
        with open(video_path, 'w') as file:
            yaml.dump(iconfig, file, allow_unicode=True)

        
        lconfig = {'lock':None,'dead':lconfig['dead']}
        with open(lock_file, 'w') as file:
            yaml.dump(lconfig, file, allow_unicode=True)

        time.sleep(5)
    
    with open(lock_file, 'r') as file:
        lconfig = yaml.safe_load(file)
        
    lconfig['dead'].append(name)

    lconfig = {'lock':lconfig['lock'],'dead':lconfig['dead']}
    with open(lock_file, 'w') as file:
        yaml.dump(lconfig, file, allow_unicode=True)



def initialize(DEG_project_path, name, inference_config, postprocessor, postprocessor_path):

    hl_deg = headless_deg(DEG_project_path)

    if inference_config!='':
        inference(hl_deg, name, inference_config, postprocessor, postprocessor_path)

            

if __name__ == '__main__':
    initialize(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
        
        
    
    
