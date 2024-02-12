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


    """
    Creates a new CBAS project using the following file structure in the working directory of the user:
    project_name
      - core
          * cameras.yaml
          - videos
      - recordings
      - models
          * models.yaml
      - postprocessors
      - figures
          - ethograms
          - rayleigh
      - performance
          * eval.yaml
          - test_sets
          - raw_values

    Parameters:
        - project_name (String): Name of the project. 
    """

    # need to grab the user's current directory
    user_dir = os.getcwd()

    # name the locations
    project = os.path.join(user_dir,project_name)

    core = os.path.join(project, 'core')
    videos = os.path.join(core, 'videos')
    frames = os.path.join(core, 'frames')

    recordings = os.path.join(project, 'recordings')

    figures = os.path.join(project, 'figures')
    ethograms = os.path.join(figures, 'ethograms')

    models = os.path.join(project, 'models')
    postprocessors = os.path.join(project, 'postprocessors')
    postprocessors_config = os.path.join(postprocessors, 'postprocessors.yaml')

    data_sets = os.path.join(project, 'data_sets')
    training_sets = os.path.join(data_sets, 'training_sets')
    test_sets = os.path.join(data_sets, 'test_sets')
    human_label = os.path.join(data_sets, 'human_labeling')

    # make sure user is not located within the main directory of a project
    project_config = os.path.join(user_dir, 'project_config.yaml')

    if os.path.exists(project_config):
        raise Exception('Cannot create a project within the main directory of another project. Move to another directory and try again.')
    else:
        project_config = os.path.join(project, 'project_config.yaml')

    # make all those directories
    os.mkdir(project)

    os.mkdir(core)
    os.mkdir(videos)
    os.mkdir(frames)
    
    os.mkdir(recordings)

    os.mkdir(figures)
    os.mkdir(ethograms)
    
    os.mkdir(models)
    os.mkdir(postprocessors)
    
    os.mkdir(data_sets)
    os.mkdir(training_sets)
    os.mkdir(test_sets)
    os.mkdir(human_label)


    # setup the yaml files
    cameras = os.path.join(core, 'cameras.yaml')
    models_config = os.path.join(models, 'models.yaml')

    # PROJECT_CONFIG
    # dumping the folder and file locations into a config file for future use

    config = {
                'project_path':project,
                'core_path':core,
                'videos_path':videos,
                'frames_path':frames,
                'recordings_path':recordings,
                'model_path':models,
                'postprocessors':postprocessors,
                'postprocessors_config':postprocessors_config,
                'figures_path':figures,
                'ethograms_path':ethograms,
                'datasets_path':data_sets,
                'trainingsets_path':training_sets,
                'testsets_path':test_sets,
                'humanlabel_path':human_label,
                'cameras_path':cameras,
                'models_path':models_config
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

    with open(models_config, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)

    config = {
                'models':[]
             }

    with open(postprocessors_config, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)


    print(f'Project creation successful! Please enter the project directory, {project}, before running any further setup commands.')

# Adds a camera to a given project
def add_camera(rtsp_url, name, project_config='undefined', safe=True):

    """
    A function for adding rtsp cameras to the project. This function does not handle the ffmpeg recording settings for video acquisition (framerate, brightness, contrast, etc.).

    Parameters:
        - rtsp_url (String): Url for the camera to be added. The user should be wary to add the camera username/password and rtsp port to the url if needed. 
        - name (String): Unique name for the camera to be added.
        - project_config (String): File location of the project config. If left as 'undefined' the user must be in the same folder as the project config yaml file.
        - safe (boolean): If True, function opens the camera url to make sure that the camera is accessible prior to adding the camera. If False, the camera url is added without any checks.
    """

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

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']
        videos = pconfig['videos_path']
        frames = pconfig['frames_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            try:
                subprocess.call(command, shell=True, timeout=5)
                print('RTSP functional!')
            except:
                raise Exception('Either the RTSP url is wrong or ffmpeg is not installed properly. You may need to include a username and password in your RTSP url if you see an authorization error. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
        
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
                raise Exception('Found a camera with this name already, please use a different name.')
            elif cam['rtsp_url'] == rtsp_url:
                raise Exception('Found an existing camera with this same rtsp url, please use a different url.')
        
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
            raise Exception('Project not found.')
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the project config file. Check for yaml syntax errors.')

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']
        videos = pconfig['videos_path']
        frames = pconfig['frames_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            try:
                subprocess.call(command, shell=True, timeout=5)
                print('RTSP functional!')
            except:
                raise Exception('Either the RTSP url is wrong or ffmpeg is not installed properly. You may need to include a username and password in your RTSP url if you see an authorization error. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
        
        # check to see if the rtsp url is functional
        if safe:
            # just grabbing the first image from the video stream
            test_frame = os.path.join(frames, 'test_frame.jpg')

            command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {test_frame}"
            subprocess.call(command, shell=True)

            if os.path.exists(test_frame):
                print('RTSP functional!')
            else:
                raise Exception('Either the RTSP url is wrong or ffmpeg is not installed properly. You may need to include a username and password in your RTSP url if you see an authorization error. You may run this function again with safe=False to generate the camera regardless of image acquisition.')
        
        
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
                raise Exception('Found a camera with this name already, please use a different name.')
            elif cam['rtsp_url'] == rtsp_url:
                raise Exception('Found an existing camera with this same rtsp url, please use a different url.')
        
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

    """
    A function for removing rtsp cameras from the project.

    Parameters:
        - name (String): Unique name for the camera to be removed.
        - project_config (String): File location of the project config. If left as 'undefined' the user must be in the same folder as the project config yaml file.
    """

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

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
        
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
            raise Exception('Could not remove the camera from the list because it could not be found.')
        
        
        
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

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

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
            raise Exception('Could not remove the camera from the list because it could not be found.')

# Displays all rtsp cameras in the project
def disp_cameras(project_config='undefined'):
    """
    A function for displaying all rtsp cameras in the project.

    Parameters:
        - project_config (String): File location of the project config. If left as 'undefined' the user must be in the same folder as the project config yaml file.
    """

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

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
        
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
        
        for cam in camera_list:
            print(cam['name'])
            print(' - URL: '+cam['rtsp_url'])
            print(' - Video Directory: '+cam['video_dir'])
            print(' - Crop Region:')
            print('   - X: '+str(cam['x']))
            print('   - Y: '+str(cam['y']))
            print('   - Width: '+str(cam['width']))
            print('   - Height: '+str(cam['height']))

        
        
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

        # grabbing the locations of the camera
        cameras = pconfig['cameras_path']

        # extract the cameras config file
        try:
            with open(cameras, 'r') as file:
                cconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

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
        
        for cam in camera_list:
            print(cam['name'])
            print(' - URL: '+cam['rtsp_url'])
            print(' - Video Directory: '+cam['video_dir'])
            print(' - Crop Region:')
            print('   - X: '+str(cam['x']))
            print('   - Y: '+str(cam['y']))
            print('   - Width: '+str(cam['width']))
            print('   - Height: '+str(cam['height']))

# Adds a model to a given project    
def add_model(model_path, model_env, name, model_type="undefined", safe=True, project_config="undefined"):

    """
    Adds a new model (DLC or DEG) for use during recordings.

    Parameters:
        - model_path (String): DEG - A path to the project folder. DLC - The path to the project config yaml.
    """

    valid_types = ['deepethogram','deeplabcut']

    if name=='lock':
        raise Exception(f"Please retry with a different name, 'lock' is used for internal things.")

    if model_type=='undefined' or model_type not in valid_types:
        raise Exception(f"Please include a valid model type. Currently available types are {str(valid_types)}.")
    
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

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']
        # postprocessors = pconfig['postprocessors_config']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        # extract the postprocessors config file
        # try:
        #     with open(postprocessors, 'r') as file:
        #         pconfig = yaml.safe_load(file)
        # except:
        #     raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        # check to see if the model is functional
        if safe:
            # check if the path exists
            if not os.path.exists(model_path):
                raise Exception('Path does not exist.')

            # split based on type
            if model_type == 'deepethogram':
                print('Processing deepethogram model...')


                env_command = f"conda activate {model_env}"

                # Run the command
                try:
                    subprocess.run(env_command, shell=True, check=True)
                    print('Successfully activated the environment!')
                except:
                    raise Exception('Error activating the environment.')

                def get_conda_env_path(env_name):
                    # Run 'conda info --envs' and capture output
                    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception("Failed to get Conda environments")

                    # Parse the output to find the desired environment
                    envs = result.stdout.splitlines()
                    env_path = None
                    for env in envs:
                        if env_name in env:
                            env_path = env.split()[-1]  # Assumes the path is the last element
                            break

                    if not env_path:
                        raise Exception(f"Environment '{env_name}' not found")

                    return env_path

                env_name = model_env
                env_base_path = get_conda_env_path(env_name)

                # Construct the path to the Python executable
                env_python_path = os.path.join(env_base_path, "python.exe")

                
                # Try finding the deepethogram project
                    
                location = os.path.dirname(os.path.abspath(__file__))


                try:
                    process = subprocess.run([env_python_path, os.path.join(location, 'deg.py'), model_path, '', '', '', ''], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE, 
                                            universal_newlines=True, 
                                            check=True)
                except subprocess.CalledProcessError as e:
                    # Print the output and error messages from the process
                    print("STDERR:", e.stderr)
                    # Re-raise the exception if needed, or handle it accordingly
                    raise Exception("Model not loading properly. Make sure this is a deepethogram model and that the correct models are set in the /project_config.yaml file of the project under 'weights:'")
                else:
                    # If there's no exception, print the output as usual
                    print(process.stdout)
                


            elif model_type == 'deeplabcut':
                print('Processing deeplabcut model.')

                env_command = f"conda activate {model_env}"

                # Run the command
                try:
                    subprocess.run(env_command, shell=True, check=True)
                    print('Successfully activated the environment!')
                except:
                    raise Exception('Error activating the environment.')

                def get_conda_env_path(env_name):
                    # Run 'conda info --envs' and capture output
                    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception("Failed to get Conda environments")

                    # Parse the output to find the desired environment
                    envs = result.stdout.splitlines()
                    env_path = None
                    for env in envs:
                        if env_name in env:
                            env_path = env.split()[-1]  # Assumes the path is the last element
                            break

                    if not env_path:
                        raise Exception(f"Environment '{env_name}' not found")

                    return env_path

                env_name = model_env
                env_base_path = get_conda_env_path(env_name)

                # Construct the path to the Python executable
                env_python_path = os.path.join(env_base_path, "python.exe")

                
                # Try finding the deepethogram project
                    
                location = os.path.dirname(os.path.abspath(__file__))


                try:
                    process = subprocess.run([env_python_path, os.path.join(location, 'dlc.py'), model_path, '', '',''], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE, 
                                            universal_newlines=True, 
                                            check=True)
                except subprocess.CalledProcessError as e:
                    # Print the output and error messages from the process
                    print("STDERR:", e.stderr)
                    # Re-raise the exception if needed, or handle it accordingly
                    raise Exception("Model not loading properly. Make sure this is a deeplabcut model and that you provided the path to the config file.")
                else:
                    # If there's no exception, print the output as usual
                    print(process.stdout)


            
        
        # assuming that the model is functional, add it as a useable model
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        # first check to see no duplicates exist
        for model in model_list:
            if model['path'] == model_path:
                raise Exception('Found a model with this path already, please use a different path.')
        
        # great, there are no conflicts. go ahead and update the models config
        model_config = {
            'name':name,
            'path':model_path,
            'postprocessor_path':None,
            'postprocessor':None,
            'env':env_name,
            'type':model_type
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
            raise Exception('Project not found.')
        
        # extract the project_config file
        try:
            with open(project_config, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the project config file. Check for yaml syntax errors.')

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']
        # postprocessors = pconfig['postprocessors_config']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
        # extract the postprocessors config file
        # try:
        #     with open(postprocessors, 'r') as file:
        #         pconfig = yaml.safe_load(file)
        # except:
        #     raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        # check to see if the model is functional
        if safe:
            # check if the path exists
            if not os.path.exists(model_path):
                raise Exception('Path does not exist.')

            # split based on type
            if model_type == 'deepethogram':
                print('Processing deepethogram model...')


                env_command = f"conda activate {model_env}"

                # Run the command
                try:
                    subprocess.run(env_command, shell=True, check=True)
                    print('Successfully activated the environment!')
                except:
                    raise Exception('Error activating the environment.')

                def get_conda_env_path(env_name):
                    # Run 'conda info --envs' and capture output
                    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception("Failed to get Conda environments")

                    # Parse the output to find the desired environment
                    envs = result.stdout.splitlines()
                    env_path = None
                    for env in envs:
                        if env_name in env:
                            env_path = env.split()[-1]  # Assumes the path is the last element
                            break

                    if not env_path:
                        raise Exception(f"Environment '{env_name}' not found")

                    return env_path

                env_name = model_env
                env_base_path = get_conda_env_path(env_name)

                # Construct the path to the Python executable
                env_python_path = os.path.join(env_base_path, "python.exe")

                
                # Try finding the deepethogram project
                    
                location = os.path.dirname(os.path.abspath(__file__))


                try:
                    process = subprocess.run([env_python_path, os.path.join(location, 'deg.py'), model_path, '', '', '', ''], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE, 
                                            universal_newlines=True, 
                                            check=True)
                except subprocess.CalledProcessError as e:
                    # Print the output and error messages from the process
                    print("STDERR:", e.stderr)
                    # Re-raise the exception if needed, or handle it accordingly
                    raise Exception("Model not loading properly. Make sure this is a deepethogram model and that the correct models are set in the /project_config.yaml file of the project under 'weights:'")
                else:
                    # If there's no exception, print the output as usual
                    print(process.stdout)
                


            elif model_type == 'deeplabcut':
                print('Processing deeplabcut model.')

                env_command = f"conda activate {model_env}"

                # Run the command
                try:
                    subprocess.run(env_command, shell=True, check=True)
                    print('Successfully activated the environment!')
                except:
                    raise Exception('Error activating the environment.')

                def get_conda_env_path(env_name):
                    # Run 'conda info --envs' and capture output
                    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                    if result.returncode != 0:
                        raise Exception("Failed to get Conda environments")

                    # Parse the output to find the desired environment
                    envs = result.stdout.splitlines()
                    env_path = None
                    for env in envs:
                        if env_name in env:
                            env_path = env.split()[-1]  # Assumes the path is the last element
                            break

                    if not env_path:
                        raise Exception(f"Environment '{env_name}' not found")

                    return env_path

                env_name = model_env
                env_base_path = get_conda_env_path(env_name)

                # Construct the path to the Python executable
                env_python_path = os.path.join(env_base_path, "python.exe")

                
                # Try finding the deepethogram project
                    
                location = os.path.dirname(os.path.abspath(__file__))


                try:
                    process = subprocess.run([env_python_path, os.path.join(location, 'dlc.py'), model_path, '', '',''], 
                                            stdout=subprocess.PIPE, 
                                            stderr=subprocess.PIPE, 
                                            universal_newlines=True, 
                                            check=True)
                except subprocess.CalledProcessError as e:
                    # Print the output and error messages from the process
                    print("STDERR:", e.stderr)
                    # Re-raise the exception if needed, or handle it accordingly
                    raise Exception("Model not loading properly. Make sure this is a deeplabcut model and that you provided the path to the config file.")
                else:
                    # If there's no exception, print the output as usual
                    print(process.stdout)


            
        
        # assuming that the model is functional, add it as a useable model
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        # first check to see no duplicates exist
        for model in model_list:
            if model['path'] == model_path:
                raise Exception('Found a model with this path already, please use a different path.')
        
        # great, there are no conflicts. go ahead and update the models config
        model_config = {
            'name':name,
            'path':model_path,
            'postprocessor_path':None,
            'postprocessor':None,
            'env':env_name,
            'type':model_type
        }
        model_list.append(model_config)
        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)

# Adds a postprocessor to a given deg model    
def add_postprocessor_to_model(name, postprocessor, project_config="undefined"):

    """
    Adds a postprocessor to a deg model for recordings.

    Parameters:
        - name (String): The name of the added deg model to add the postprocessor to.
    """
    
    
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

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']
        postprocessors = pconfig['postprocessors_config']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        # extract the postprocessors config file
        try:
            with open(postprocessors, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            
        
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        index = 0
        for model in model_list:
            if model['name'] == name:
                break
            index+=1
        
        model_list[index]['postprocessor_path'] = postprocessors
        model_list[index]['postprocessor'] = postprocessor

        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)
        
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

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']
        postprocessors = pconfig['postprocessors_config']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
        # extract the postprocessors config file
        try:
            with open(postprocessors, 'r') as file:
                pconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')

        
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        index = 0
        for model in model_list:
            if model['name'] == name:
                break
            index+=1
        
        model_list[index]['postprocessor_path'] = postprocessors
        model_list[index]['postprocessor'] = postprocessor
        
        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)
    
    print(f'Postprocessor successfully added to {name}')

# removes a model to a given project    
def remove_model(name, project_config="undefined"):

    """
    Removes an added model from a project.

    Parameters:
        - name (String): The name of the added model to remove.
    """
    
    
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

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            
        
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        index = 0
        found = False
        for model in model_list:
            if model['name'] == name:
                found = True
                break
            index+=1
        if not found:
            raise Exception('Model not found.')
            
        model_list.pop(index)

        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)
        
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

        # grabbing the locations of the models yaml file
        models = pconfig['models_path']

        # extract the models config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the cameras config file. Check for yaml syntax errors.')
            
        
        model_list = mconfig['models']
        if model_list is None:
            model_list = []
        
        index = 0
        found = False
        for model in model_list:
            if model['name'] == name:
                found = True
                break
            index+=1
        if not found:
            raise Exception('Model not found.')

        model_list.pop(index)
        
        new_config = {
            'models':model_list
        }

        with open(models, 'w+') as file:
            yaml.dump(new_config, file, allow_unicode=True)

    print(f'Successfully removed {name}')

