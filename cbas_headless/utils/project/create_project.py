import time
import ctypes
import yaml
import os
import math
import sys
import subprocess

def create_project(project_name):
    # creates a new CBAS project using the following file structure:
    # project_name
    #   - core
    #       * cameras.yaml
    #       - videos
    #   - recordings
    #   - figures
    #       - model
    #           * models.yaml
    #       - ethograms
    #       - rayleigh
    #   - performance
    #       * eval.yaml
    #       - test_sets
    #       - raw_values

    # need to grab the user's current directory
    user_dir = os.getcwd()

    # name the locations
    project = os.path.join(user_dir,project_name)

    core = os.path.join(project, 'core')
    videos = os.path.join(core, 'videos')
    frames = os.path.join(core, 'frames')

    recordings = os.path.join(project, 'recordings')

    figures = os.path.join(project, 'figures')
    model = os.path.join(figures, 'model')
    ethograms = os.path.join(figures, 'ethograms')
    rayleigh = os.path.join(figures, 'rayleigh')

    performance = os.path.join(project, 'performance')
    test_sets = os.path.join(performance, 'test_sets')
    raw_values = os.path.join(performance, 'raw_values')

    # make sure user is not located within the main directory of a project
    project_config = os.path.join(user_dir, 'project_config.yaml')

    if os.path.exists(project_config):
        print('Cannot create a project within the main directory of another project. Move to another directory and try again.')
        exit()
    else:
        project_config = os.path.join(project, 'project_config.yaml')

    # make all those directories
    os.mkdir(project)

    os.mkdir(core)
    os.mkdir(videos)
    os.mkdir(frames)
    
    os.mkdir(recordings)

    os.mkdir(figures)
    os.mkdir(model)
    os.mkdir(ethograms)
    os.mkdir(rayleigh)
    
    os.mkdir(performance)
    os.mkdir(test_sets)
    os.mkdir(raw_values)

    # setup the yaml files
    cameras = os.path.join(core, 'cameras.yaml')
    models = os.path.join(model, 'models.yaml')
    eval = os.path.join(performance, 'eval.yaml')

    # PROJECT_CONFIG
    # dumping the folder and file locations into a config file for future use

    config = {
                'project_path':project,
                'core_path':core,
                'videos_path':videos,
                'frames_path':frames,
                'recordings_path':recordings,
                'figures_path':figures,
                'model_path':model,
                'ethograms_path':ethograms,
                'rayleigh_path':rayleigh,
                'performance_path':performance,
                'test_sets_path':test_sets,
                'raw_values_path':raw_values,
                'cameras_path':cameras,
                'models_path':models,
                'eval_path':eval,
            }

    with open(project_config, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)

    # CAMERAS
    # contrast ranges from 0-2
    # brightness ranges from -1-1
    # framerate drastically increases file size, set to 10 fps to begin with
    # scale indicates the rescaled size of the final square image
    #
    # cameras
    #   - cam1
    #       - rtsp_url
    #       - width
    #       - height
    #       - x
    #       - y

    config = {
                'contrast':1,
                'brightness':0,
                'framerate':10,
                'scale':256,
                'cameras':[]
            }

    with open(cameras, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)

    # MODELS
    # deg_path indicates the system path to the deepethogram model that is to be used
    # dlc_path indicates the system path to the deeplabcut model that is to be used

    config = {
                'deg_path':'',
                'dlc_path':''
            }

    with open(models, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)

    # EVAL
    # window_size determines the timescale at which to grade the model, default is 1 frame
    # metrics to be included, default is precision, recall, f1 score, specificity, balanced accuracy, and normalized matthews correlation coefficient 

    config = {
                'window_size':1,
                'precision':True,
                'recall':True,
                'f1_score':True,
                'specificity':True,
                'balanced_accuracy':True,
                'normalized_matthews_cc':True
            }

    with open(eval, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)


def add_camera(rtsp_url, project_config='undefined', safe=True):

    if project_config=='undefined':
        # assume that the user is located in an active project
        user_dir = os.getcwd()

        # make sure user is located within the main directory of a project
        project_config = os.path.join(user_dir, 'project_config.yaml')

        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit()
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit()

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']
        videos = pconfig['videos_path']
        frames = pconfig['frames_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            exit()

        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            subprocess.call(command, shell=True)

            if os.path.exists(test_frame):
                print('RTSP functional!')
            else:
                print('Either the RTSP url is wrong or ffmpeg is not installed properly. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
                exit()
        
    else:
        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit()
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit()

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']
        videos = pconfig['videos_path']
        frames = pconfig['frames_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            exit()

        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            subprocess.call(command, shell=True)

            if os.path.exists(test_frame):
                print('RTSP functional!')
            else:
                print('Either the RTSP url is wrong or ffmpeg is not installed properly. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
                exit()

        
