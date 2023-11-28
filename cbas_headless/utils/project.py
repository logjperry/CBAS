import time
import ctypes
import yaml
import os
import math
import sys
from sys import exit
import subprocess
import shutil

# Intializes a new project
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
        exit(0)
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
    #   - name
    #   - rtsp_url
    #   - video_dir
    #   - width
    #   - height
    #   - x
    #   - y

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
    # model_paths indicate the system paths to the machine learning models that are to be used

    config = {
                'models':[]
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

# Adds a camera to a given project
def add_camera(rtsp_url, name, project_config='undefined', safe=True):

    if project_config=='undefined':
        # assume that the user is located in an active project
        user_dir = os.getcwd()

        # make sure user is located within the main directory of a project
        project_config = os.path.join(user_dir, 'project_config.yaml')

        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit(0)
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit(0)

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
            exit(0)

        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            subprocess.call(command, shell=True)

            if os.path.exists(test_frame):
                print('RTSP functional!')
            else:
                print()
                print('Either the RTSP url is wrong or ffmpeg is not installed properly. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
                print('You may need to include a username and password in your RTSP url if you see an authorization error.')
                print()
                exit(0)
        
        # assuming that the rtsp url is functional, make a dedicated location for storing videos and setup the config
        camera_list = cconfig['cameras']

        # REMINDER
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y
        
        # first check to see no duplicates exist
        for cam in camera_list:
            if cam['name'] == name:
                print('Found a camera with this name already, please use a different name.')
                exit(0)
            elif cam['rtsp_url'] == rtsp_url:
                print('Found an existing camera with this same rtsp url, please use a different url.')
                exit(0)
        
        # great, there are no conflicts. go ahead and make a directory for future videos and a camera config for this new camera
        video_folder = os.path.join(videos,name)
        os.mkdir(video_folder)
        new_config = {
            'name':name,
            'rtsp_url':rtsp_url,
            'video_dir':video_folder,
            'width':1,
            'height':1,
            'x':0,
            'y':0
        }

        # ok, now to save it all
        camera_list.append(new_config)

        # CAMERAS
        # contrast ranges from 0-2
        # brightness ranges from -1-1
        # framerate drastically increases file size, set to 10 fps to begin with
        # scale indicates the rescaled size of the final square image
        #
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y

        config = {
                    'contrast':cconfig['contrast'],
                    'brightness':cconfig['brightness'],
                    'framerate':cconfig['framerate'],
                    'scale':cconfig['scale'],
                    'cameras':camera_list
                }

        with open(cameras, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)
        
    else:
        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit(0)
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit(0)

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
            exit(0)

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
                exit(0)

        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            subprocess.call(command, shell=True)

            if os.path.exists(test_frame):
                print('RTSP functional!')
            else:
                print()
                print('Either the RTSP url is wrong or ffmpeg is not installed properly. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
                print('You may need to include a username and password in your RTSP url if you see an authorization error.')
                print()
                exit(0)
        
        # assuming that the rtsp url is functional, make a dedicated location for storing videos and setup the config
        camera_list = cconfig['cameras']

        # REMINDER
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y
        
        # first check to see no duplicates exist
        for cam in camera_list:
            if cam['name'] == name:
                print('Found a camera with this name already, please use a different name.')
                exit(0)
            elif cam['rtsp_url'] == rtsp_url:
                print('Found an existing camera with this same rtsp url, please use a different url.')
                exit(0)
        
        # great, there are no conflicts. go ahead and make a directory for future videos and a camera config for this new camera
        video_folder = os.path.join(videos,name)
        os.mkdir(video_folder)
        new_config = {
            'name':name,
            'rtsp_url':rtsp_url,
            'video_dir':video_folder,
            'width':1,
            'height':1,
            'x':0,
            'y':0
        }

        # ok, now to save it all
        camera_list.append(new_config)

        # CAMERAS
        # contrast ranges from 0-2
        # brightness ranges from -1-1
        # framerate drastically increases file size, set to 10 fps to begin with
        # scale indicates the rescaled size of the final square image
        #
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y

        config = {
                    'contrast':cconfig['contrast'],
                    'brightness':cconfig['brightness'],
                    'framerate':cconfig['framerate'],
                    'scale':cconfig['scale'],
                    'cameras':camera_list
                }

        with open(cameras, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)

# Removes a camera from a given project, scorched earth style
def remove_camera(name, project_config='undefined'):

    if project_config=='undefined':
        # assume that the user is located in an active project
        user_dir = os.getcwd()

        # make sure user is located within the main directory of a project
        project_config = os.path.join(user_dir, 'project_config.yaml')

        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit(0)
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit(0)

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            exit(0)
        
        # just remove everything related to the named camera
        camera_list = cconfig['cameras']

        # REMINDER
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y
        
        # find the named camera
        new_camera_list = []
        found = False
        for cam in camera_list:
            if cam['name'] == name:
                found = True
                print('Removing the named camera.')
                # great, found it. go ahead and remove the video folder and the camera config
                video_folder = cam['video_dir']
                shutil.rmtree(video_folder)
            else:
                new_camera_list.append(cam)

        # CAMERAS
        # contrast ranges from 0-2
        # brightness ranges from -1-1
        # framerate drastically increases file size, set to 10 fps to begin with
        # scale indicates the rescaled size of the final square image
        #
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y

        config = {
                    'contrast':cconfig['contrast'],
                    'brightness':cconfig['brightness'],
                    'framerate':cconfig['framerate'],
                    'scale':cconfig['scale'],
                    'cameras':new_camera_list
                }

        with open(cameras, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)
                
        if not found:
            print('Could not remove the camera from the list because it could not be found.')
        
        
        
    else:
        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit(0)
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit(0)

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            exit(0)

        # just remove everything related to the named camera
        camera_list = cconfig['cameras']

        # REMINDER
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y
        
        # find the named camera
        new_camera_list = []
        found = False
        for cam in enumerate(camera_list):
            if cam['name'] == name:
                found = True
                print('Removing the named camera.')
                # great, found it. go ahead and remove the video folder and the camera config
                video_folder = cam['video_dir']
                shutil.rmtree(video_folder)
            else:
                new_camera_list.append(cam)

        # CAMERAS
        # contrast ranges from 0-2
        # brightness ranges from -1-1
        # framerate drastically increases file size, set to 10 fps to begin with
        # scale indicates the rescaled size of the final square image
        #
        # cameras
        #   - name
        #   - rtsp_url
        #   - video_dir
        #   - width
        #   - height
        #   - x
        #   - y

        config = {
                    'contrast':cconfig['contrast'],
                    'brightness':cconfig['brightness'],
                    'framerate':cconfig['framerate'],
                    'scale':cconfig['scale'],
                    'cameras':new_camera_list
                }

        with open(cameras, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)
                
        if not found:
            print('Could not remove the camera from the list because it could not be found.')

        
        
def add_model(model_path, name, type="undefined", safe=True, project_config="undefined"):

    valid_types = ['deepethogram','deeplabcut']

    if type=='undefined' or type not in valid_types:
        print(f"Please include a valid model type. Currently available types are {str(valid_types)}.")
        exit(0)
    
    if project_config=='undefined':
        # assume that the user is located in an active project
        user_dir = os.getcwd()

        # make sure user is located within the main directory of a project
        project_config = os.path.join(user_dir, 'project_config.yaml')

        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit(0)
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit(0)

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            exit(0)

        # check to see if the model is functional
        if safe:
            # check if the path exists
            if not os.path.exists(model_path):
                print('Path does not exists.')
                exit(0)

            # split based on type
            if type == 'deepethogram':
                print('Processing deepethogram model.')
            elif type == 'deeplabcut':
                print('Processing deeplabcut model.')
            
        
        # assuming that the model is functional, add it as a useable model
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        # first check to see no duplicates exist
        for model in model_list:
            if model['path'] == model_path:
                print('Found a model with this path already, please use a different path.')
                exit(0)
        
        # great, there are no conflicts. go ahead and update the models config
        model_config = {
            'name':name,
            'path':model_path,
            'type':type
        }
        model_list.append(model_config)
        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)
        
    else:
        if os.path.exists(project_config):
            print('Project found.')
        else:
            print('Project not found.')
            exit(0)
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the project config file. Check for yaml syntax errors.')
            exit(0)

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            print('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            exit(0)

        # check to see if the model is functional
        if safe:
            # check if the path exists
            if not os.path.exists(model_path):
                print('Path does not exists.')
                exit(0)

            # split based on type
            if type == 'deepethogram':
                print('Processing deepethogram model.')
            elif type == 'deeplabcut':
                print('Processing deeplabcut model.')
            
        
        # assuming that the model is functional, add it as a useable model
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        # first check to see no duplicates exist
        for model in model_list:
            if model['path'] == model_path:
                print('Found a model with this path already, please use a different path.')
                exit(0)
        
        # great, there are no conflicts. go ahead and update the models config
        model_config = {
            'name':name,
            'path':model_path,
            'type':type
        }
        model_list.append(model_config)
        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)