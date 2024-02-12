import os
import subprocess
from multiprocessing import Process
import yaml
from sys import exit
import psutil
from datetime import datetime
import time
import signal
import glob
import shutil
import json


def storage_orchestrator(recordingConfig):

    # the failure threshold, try to keep the ball rolling for x attempts, for non-critical things give much fewer attempts
    attempts = 100
    not_critical_attempts = 5

    # figure out which cameras to watch
    success = False 
    trys = 0
    while not success:
        try:
            with open(recordingConfig, 'r') as file:
                rconfig = yaml.safe_load(file)
                cameras = rconfig['cameras']
            success = True
        except Exception as e:
            success = False
            time.sleep(1)
            trys+=1

            if trys>attempts:
                print(e.message, e.args)
                raise Exception('Exceeded the number of attempts, exiting storage orchestrator thread.')


    cams_to_watch = cameras

    # get the recording folder, for the basic orchestrator, store videos in a general videos folder
    recordingFolder = os.path.join(os.path.split(recordingConfig)[0], 'videos')

    # check to see if it exists already... it shouldn't
    if not os.path.isdir(recordingFolder):
        os.mkdir(recordingFolder)
    
    # make the subfolders for each camera
    for camera in cams_to_watch:
        foldername = os.path.join(recordingFolder, camera['name'])
        if not os.path.isdir(foldername):
            os.mkdir(foldername)

    store_last = []

    # loop indefinitely, check for videos that have finished recording. push done videos to destination folder
    while True:
        
        for camera in cams_to_watch:


            # grab the videos that have been recorded from the core/videos/camera location
            video_dir = camera['video_dir']

            camera_name = camera['name']
            
            # check the camera config to see if finished, add to store last if it is
            success = False 
            trys = 0
            while not success:
                try:
                    with open(recordingConfig, 'r') as file:
                        rconfig = yaml.safe_load(file)
                    
                        if rconfig['cameras_time'][camera_name]['end_time'] is not None:
                            if camera_name not in store_last:
                                store_last.append(camera_name)

                    success = True
                except Exception as e:
                    success = False
                    time.sleep(1)
                    trys+=1

                    if trys>attempts:
                        print(e.message, e.args)
                        raise Exception('Exceeded the number of attempts, exiting storage orchestrator thread.')

            real_videos = []

            files = glob.glob(os.path.join(video_dir, '*'))
            for file in files:
                if os.path.isfile(file):
                    real_videos.append(file)
            
            if len(real_videos)==0:
                continue

            real_videos.sort()

            if camera_name not in store_last:
                # Don't include the last one
                real_videos = real_videos[:-1]


            finished_videos = []

            # find the finished video names
            subfoldername = os.path.join(recordingFolder, camera['name'])
            for subdir, dirs, files in os.walk(subfoldername):
                for file in files:
                    finished_videos.append(os.path.join(video_dir,file))
            
            # discard videos that have already finished from this list
            real_videos = [vid for vid in real_videos if vid not in finished_videos]


            for video in real_videos:
                video_name = os.path.split(video)[1]
                dest = os.path.join(subfoldername,video_name)

                local_time = time.localtime(os.path.getctime(video))
                original_ctime = time.strftime("%H:%M:%S", local_time)

                # try our best to move the videos
                success = False 
                trys = 0
                while not success:
                    if not os.path.exists(dest):
                        try:
                            shutil.copy2(video, dest)
                            success = True
                        except Exception as e:
                            success = False
                            time.sleep(1)
                            trys+=1

                            if trys>not_critical_attempts:
                                print('Tried to move the files, gave it our best shot for now.')
                                success = True
                                break
                    else:
                        success = True
                
                # try our best to remove and record the files
                success = False 
                trys = 0
                while not success:
                    if os.path.exists(dest):
                    
                        try:
                            if os.path.exists(video):
                                os.remove(video)

                            # update the camera config
                            with open(recordingConfig, 'r') as file:
                                rconfig = yaml.safe_load(file)
                                rconfig['cameras_files'][camera_name][video_name] = original_ctime

                            with open(recordingConfig, 'w+') as file:
                                yaml.dump(rconfig, file, allow_unicode=True)

                            success = True
                        except Exception as e:
                            success = False
                            time.sleep(1)
                            trys+=1

                            if trys>not_critical_attempts:
                                print('Tried to move the files, gave it our best shot for now.')
                                success = True
                                break
                    else:
                        success = True

        
        time.sleep(1)

def kill_subprocesses(dict, keys):
    for k in keys:
        try:
            dict[k].kill()
        except:
            continue

def inference_orchestrator(recordingConfig, modelConfigs):

    # this method has a TON of error handling, and for good reason
    # any error here will lead to zombie processes and failed inferences

    loaded_models = {}

    # the failure threshold, try to keep the ball rolling for x attempts, for non-critical things give much fewer attempts
    attempts = 100
    not_critical_attempts = 5


    # figure out which cameras to watch
    success = False 
    trys = 0
    while not success:
        try:
            with open(recordingConfig, 'r') as file:
                rconfig = yaml.safe_load(file)
                models = rconfig['cameras_per_model'].keys()
            success = True
        except Exception as e:
            success = False
            time.sleep(1)
            trys+=1

            if trys>attempts:
                print(e.message, e.args)
                raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')
    

    # Load the relevant models
    models_in_use = []
    for m in models:
        
        if len(rconfig['cameras_per_model'][m]) == 0:
            continue 

        model = None
        for m1 in modelConfigs:
            if m1['name']==m:
                model = m1
        if model==None:
            continue 
        else:

            models_in_use.append(m)

            menv = model['env']
            mtype = model['type']
            mpath = model['path']
            pname = model['postprocessor']
            ppath = model['postprocessor_path']


            if mtype == 'deepethogram':
                
                env_command = f"conda activate {menv}"

                # Run the command
                try:
                    subprocess.run(env_command, shell=True, check=True)
                    print('Successfully activated the environment!')
                except:
                    kill_subprocesses(loaded_models, loaded_models.keys())
                    raise Exception('Error activating the environment.')

                def get_conda_env_path(env_name):
                    # Run 'conda info --envs' and capture output
                    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                    if result.returncode != 0:
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception("Failed to get Conda environments")

                    # Parse the output to find the desired environment
                    envs = result.stdout.splitlines()
                    env_path = None
                    for env in envs:
                        if env_name in env:
                            env_path = env.split()[-1]  # Assumes the path is the last element
                            break

                    if not env_path:
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception(f"Environment '{env_name}' not found")

                    return env_path

                env_name = menv
                env_base_path = get_conda_env_path(env_name)

                # Construct the path to the Python executable
                env_python_path = os.path.join(env_base_path, "python.exe")

                # Try finding the deepethogram file in CBAS
                location = os.path.dirname(os.path.abspath(__file__))
                
                storageFolder = os.path.join(os.path.split(recordingConfig)[0], 'videos')

                pipe = os.path.join(storageFolder, m+'.yaml')


                try:
                    if pname is None:
                        process = subprocess.Popen([env_python_path, os.path.join(location, 'deg.py'), mpath, m, pipe, '', ''])
                    else:
                        process = subprocess.Popen([env_python_path, os.path.join(location, 'deg.py'), mpath, m, pipe, pname, ppath])
                    loaded_models[m] = process
                except subprocess.CalledProcessError as e:
                    # Re-raise the exception if needed, or handle it accordingly
                    kill_subprocesses(loaded_models, loaded_models.keys())
                    raise Exception("Model not loading properly. Make sure this is a deepethogram model and that the correct models are set in the /project_config.yaml file of the project under 'weights:'")


            elif mtype == 'deeplabcut':
                

                env_command = f"conda activate {menv}"

                # Run the command
                try:
                    subprocess.run(env_command, shell=True, check=True)
                    print('Successfully activated the environment!')
                except:
                    kill_subprocesses(loaded_models, loaded_models.keys())
                    raise Exception('Error activating the environment.')

                def get_conda_env_path(env_name):
                    # Run 'conda info --envs' and capture output
                    result = subprocess.run(['conda', 'info', '--envs'], capture_output=True, text=True)
                    if result.returncode != 0:
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception("Failed to get Conda environments")

                    # Parse the output to find the desired environment
                    envs = result.stdout.splitlines()
                    env_path = None
                    for env in envs:
                        if env_name in env:
                            env_path = env.split()[-1]  # Assumes the path is the last element
                            break

                    if not env_path:
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception(f"Environment '{env_name}' not found")

                    return env_path

                env_name = menv
                env_base_path = get_conda_env_path(env_name)

                # Construct the path to the Python executable
                env_python_path = os.path.join(env_base_path, "python.exe")

                
                # Try finding the deepethogram project
                location = os.path.dirname(os.path.abspath(__file__))

                
                storageFolder = os.path.join(os.path.split(recordingConfig)[0], 'videos')
                modelFolder = os.path.join(os.path.split(recordingConfig)[0], m)

                pipe = os.path.join(storageFolder, m+'.yaml')


                try:
                    process = subprocess.Popen([env_python_path, os.path.join(location, 'dlc.py'), mpath, m, pipe, modelFolder])
                    loaded_models[m] = process
                except subprocess.CalledProcessError as e:
                    # Re-raise the exception if needed, or handle it accordingly
                    kill_subprocesses(loaded_models, loaded_models.keys())
                    raise Exception("Model not loading properly. Make sure this is a deeplabcut model and that you provided the path to the config file.")


    pipe_set = []
    global_lock = os.path.join(os.path.split(recordingConfig)[0], 'videos','lock.yaml')
    lconfig = {'lock':None, 'dead':[]}

    # initialize the global lock
    success = False 
    trys = 0
    while not success:
        try:
            with open(global_lock, 'w') as file:
                yaml.dump(lconfig, file, allow_unicode=True)
            success = True
        except Exception as e:
            success = False
            time.sleep(1)
            trys+=1

            if trys>attempts:
                print(e.message, e.args)
                kill_subprocesses(loaded_models, loaded_models.keys())
                raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')



    # loop indefinitely, check for videos that have finished recording. push done videos to destination folder
    while True:

        # we want to make sure that absolutely all of the models are done inferring as many videos as possible
        # absolute_finish is a stopping condition so that this thread only ends when no videos are 'ready' for inference by any model
        absolute_finish = True
        # wu is a stopping condition so that this thread knows the recording has been terminated by the user
        wu = False

        for m in models_in_use:

            # figure out which cameras to watch depending on the model to allow inferencing next
            success = False 
            trys = 0
            while not success:
                try:
                    with open(recordingConfig, 'r') as file:
                        rconfig = yaml.safe_load(file)
                        cameras = rconfig['cameras']
                        cams_per_model = rconfig['cameras_per_model'][m]
                    success = True
                except Exception as e:
                    success = False
                    time.sleep(1)
                    trys+=1

                    if trys>attempts:
                        print(e.message, e.args)
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')

            cams_to_watch = [cam for cam in cameras if cam['name'] in cams_per_model]

            # there are no cameras for this model to infer, take a few seconds and continue with other models
            if len(cams_to_watch)==0:
                time.sleep(10)
                continue

            # get the video storage folder, pipe, and model output folder
            storageFolder = os.path.join(os.path.split(recordingConfig)[0], 'videos')
            pipe = os.path.join(storageFolder, m+'.yaml')
            modelFolder = os.path.join(os.path.split(recordingConfig)[0], m)

            # check to see if the model output folder exists already, create it if not
            if not os.path.isdir(modelFolder):
                os.mkdir(modelFolder)
            
            # these arrays store the videos that have yet to be inferred and those that are finished
            ready = []
            finished = []

            wrapup = False 

            # try our best to open or initialize the model pipe and to grab the global lock
            success = False 
            trys = 0
            while not success:
                try:
                    if os.path.exists(pipe):
                        with open(pipe, 'r') as file:
                            iconfig = yaml.safe_load(file)
                            finished = iconfig['finished']
                            wrapup = iconfig['wrapup']
                    else:
                        # initialize the pipe
                        iconfig = {'ready':ready,'finished':finished,'wrapup':False,'lock':global_lock, 'avg_times':[], 'times':[], 'iter':0}
                        with open(pipe, 'w') as file:
                            yaml.dump(iconfig, file, allow_unicode=True)
                    

                    with open(global_lock, 'r') as file:
                        lconfig = yaml.safe_load(file)
                        lock = lconfig['lock']
                        dead = lconfig['dead']

                    success = True
                except Exception as e:
                    success = False
                    time.sleep(1)
                    trys+=1

                    if trys>attempts:
                        print(e.message, e.args)
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')

            # count the dead models, if all are dead, kill this thread. if only the current model is dead, move on
            if m in dead:
                done = True
                for m1 in models_in_use:
                    if m1 not in dead:
                        done = False
                if done:
                    absolute_finish = True
                    wu = True
                    break 
                else:
                    continue

            # this is where we will wait until the current model is finished with the lock
            if lock != None:

                # this is critical code, and hanging is definitely possible. we try to mitigate that by doing the 'dead' check on the model with the lock 
                done = False
                while not done:
                    time.sleep(5)
                    
                    # grab the global lock and check if we are ready to pass it to the next model
                    success = False 
                    trys = 0
                    while not success:
                        try:
                            with open(global_lock, 'r') as file:
                                lconfig = yaml.safe_load(file)
                                lock = lconfig['lock']
                                dead = lconfig['dead']
                            success = True
                        except Exception as e:
                            success = False
                            time.sleep(1)
                            trys+=1

                            if trys>attempts:
                                print(e.message, e.args)
                                kill_subprocesses(loaded_models, loaded_models.keys())
                                raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')

                    # if the lock is ours, we can continue
                    if lock==None:
                        done = True
                    else:
                        try:
                            
                            poll = loaded_models[lock].poll()
                            if poll is None:
                                done = False
                            else:
                                # the model with the lock died but didn't inform us
                                done = True 
                                # we should add the model to the list of dead models
                                success = False 
                                trys = 0
                                while not success:
                                    try:
                                        with open(global_lock, 'r') as file:
                                            lconfig = yaml.safe_load(file)
                                            dead = lconfig['dead']

                                        # make sure that we aren't double adding it
                                        if lock not in dead:
                                            lconfig['dead'].append(lock)
                                        lconfig['lock'] = None

                                        with open(global_lock, 'w') as file:
                                            yaml.dump(lconfig, file, allow_unicode=True)

                                        success = True
                                    except Exception as e:
                                        success = False
                                        time.sleep(1)
                                        trys+=1

                                        if trys>attempts:
                                            print(e.message, e.args)
                                            kill_subprocesses(loaded_models, loaded_models.keys())
                                            raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')
                                break
                        except:
                            # model polling failed, that is really bad
                            # go ahead and kill the child process and then try again
                            loaded_models[lock].kill()

            # now we are attempting to pass the lock to the next model!
            # first do another check to make sure the model isn't dead
            success = False 
            trys = 0
            while not success:
                try:
                    with open(global_lock, 'r') as file:
                        lconfig = yaml.safe_load(file)
                        dead = lconfig['dead']
                    success = True
                except Exception as e:
                    success = False
                    time.sleep(1)
                    trys+=1

                    if trys>attempts:
                        print(e.message, e.args)
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')
            
            # count the dead models, if all are dead, kill this thread. if only the current model is dead, move on
            if m in dead:
                done = True
                for m1 in models_in_use:
                    if m1 not in dead:
                        done = False
                if done:
                    absolute_finish = True
                    wu = True
                    break 
                else:
                    continue

            # ok, refresh the crucial parameters from the pipe
            success = False 
            trys = 0
            while not success:
                try:
                    with open(pipe, 'r') as file:
                        iconfig = yaml.safe_load(file)
                        finished = iconfig['finished']
                        wrapup = iconfig['wrapup']
                    success = True
                except Exception as e:
                    success = False
                    time.sleep(1)
                    trys+=1

                    if trys>attempts:
                        print(e.message, e.args)
                        kill_subprocesses(loaded_models, loaded_models.keys())
                        raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')
            
            # now we can do the fun stuff, add the videos that are ready to be inferenced
            for camera in cams_to_watch:

                camera_name = camera['name']

                video_dir = os.path.join(storageFolder, camera_name)

                real_videos = []

                files = glob.glob(os.path.join(video_dir, '*'))
                for file in files:
                    if os.path.isfile(file):
                        real_videos.append(file)
                
                
                if len(real_videos)==0:
                    continue

                real_videos.sort()


                finished_videos = []
                for video in finished:

                    source_folder = video
                    video_name = os.path.split(video)[1]
                    dest_folder = os.path.join(modelFolder, video_name)

                    # check to see that the inference files are stored correctly, if not, move them to the right place
                    if os.path.exists(dest_folder):
                        finished_videos.append(video_name+'.mp4')
                    else:
                        os.mkdir(dest_folder)
                        done_files = glob.glob(os.path.join(source_folder, '*'))
                        move = []
                        for f in done_files:
                            if os.path.isfile(f) and os.path.splitext(f)[1]!='.mp4':
                                move.append(f)

                        for f in move:

                            dest = os.path.join(dest_folder, os.path.split(f)[1])
                            
                            # giving the files the chance to get moved to the new location
                            success = False 
                            trys = 0
                            while not success:
                                try:
                                    shutil.copy2(f, dest)
                                    success = True
                                except Exception as e:
                                    success = False
                                    time.sleep(1)
                                    trys+=1

                                    if trys>not_critical_attempts:
                                        print('Tried to move the files, gave it our best shot for now.')
                                        success = True
                                        break

                            # try our best to remove the files
                            success = False 
                            trys = 0
                            while not success:
                                if os.path.exists(dest):
                                    try:
                                        os.remove(f)
                                        success = True
                                    except Exception as e:
                                        success = False
                                        time.sleep(1)
                                        trys+=1

                                        if trys>not_critical_attempts:
                                            print('Tried to remove the files, gave it our best shot for now.')
                                            success = True
                                            break
                                else:
                                    success = True


                        # try our best to remove the old folder
                        success = False 
                        trys = 0
                        while not success:
                            if os.path.exists(source_folder):
                                try:
                                    shutil.rmtree(source_folder)
                                    success = True
                                except Exception as e:
                                    success = False
                                    time.sleep(1)
                                    trys+=1

                                    if trys>not_critical_attempts:
                                        print('Tried to remove the old folder, gave it our best shot for now.')
                                        success = True
                                        break
                            else:
                                success = True
                        
                        finished_videos.append(video_name+'.mp4')
                
                
                # discard videos that have already finished from this list
                real_videos = [vid for vid in real_videos if os.path.split(vid)[1] not in finished_videos]


                # queue up the videos that are ready to be inferenced
                for rv in real_videos:
                    ready.append(rv)


                

            # set up the config file with the files to to inferenced and pass the lock to the next model
            iconfig = {'ready':ready,'finished':finished,'wrapup':wrapup,'lock':global_lock, 'avg_times':iconfig['avg_times'], 'times':iconfig['times'], 'iter':iconfig['iter']}

            if len(ready)>0:
                absolute_finish = False

                # try our best to write everything to the pipe and to the global lock
                success = False 
                trys = 0
                while not success:
                    try:
                        with open(pipe, 'w') as file:
                            yaml.dump(iconfig, file, allow_unicode=True)

                        lconfig = {'lock':m, 'dead':lconfig['dead']}
                        with open(global_lock, 'w') as file:
                            yaml.dump(lconfig, file, allow_unicode=True)
                        success = True
                    except Exception as e:
                        success = False
                        time.sleep(1)
                        trys+=1

                        if trys>attempts:
                            print(e.message, e.args)
                            kill_subprocesses(loaded_models, loaded_models.keys())
                            raise Exception('Exceeded the number of attempts, exiting orchestrator thread.')

            if wrapup:
                wu = True
                

            time.sleep(5)


        if absolute_finish and wu:
            kill_subprocesses(loaded_models, loaded_models.keys())
            break
            


