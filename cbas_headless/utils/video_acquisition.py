import tkinter as tk
from tkinter import filedialog, Button, StringVar, Checkbutton
from PIL import Image, ImageTk
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
import orchestrate

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time

class ImageCropTool:
    def __init__(self, root, images, cconfig=None):
        # Initialize the main window and canvas for the tool
        self.root = root
        self.root.title('Image Crop Tool')

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.terminated = False


        # Bind mouse events for drawing the rectangle
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)

        # Button for saving the region of interest (ROI)
        self.crop_button = Button(root, text="Save ROI", command=self.crop_image)
        self.crop_button.pack(pady=(20,30))

        # Variables to store image information and cropping coordinates
        self.filename = None
        self.rect = None
        self.start_x = None
        self.start_y = None
        self.submitted = False 
        self.region = []
        self.regions = []
        self.images = images
        if cconfig!=None:
            self.cam_list = cconfig['cameras'].copy()
        else:
            self.cam_list = None

        # Process images for cropping
        self.getImageCropRegions(images)

    def close_gracefully(self):
        self.terminated = True
        self.root.destroy()

    def resize_image(self, image, width, height):
        # Resize image maintaining aspect ratio
        img_aspect = image.width / image.height
        win_aspect = width / height

        if img_aspect > win_aspect:
            return image.resize((width, int(width/img_aspect)), Image.ANTIALIAS)
        else:
            return image.resize((int(height*img_aspect), height), Image.ANTIALIAS)

    def on_press(self, event):
        # Start drawing the rectangle from where the mouse is clicked
        self.canvas.delete(self.rect)
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")
        

    def on_drag(self, event):
        # Adjust the rectangle dimensions as the mouse is dragged
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def crop_image(self):
        # Function to save the cropped region
        if not self.rect:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.rect)

        x1 = x1-20
        y1 = y1-20
        x2 = x2-20
        y2 = y2-20

        # Calculate normalized values for the crop region
        norm_width = (x2 - x1) / self.image.width
        norm_height = (y2 - y1) / self.image.height
        norm_x_origin = x1 / self.image.width
        norm_y_origin = y1 / self.image.height

        self.region = [norm_width, norm_height, norm_x_origin, norm_y_origin]

        self.submitted = True
        self.save_region()

        # Uncomment the following lines to save the cropped image to a file
        # cropped = self.image.crop((x1, y1, x2, y2))
        # cropped.save("cropped.png")
    
    def getImageCropRegions(self, im_list):
        # Function to handle multiple images for cropping
        if hasattr(self, "image_on_canvas"):
            self.canvas.delete(self.image_on_canvas)
        
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        print(im_list[0])

        if not os.path.exists(im_list[0]):
            raise Exception('Error loading frame image.')

        self.image = Image.open(im_list[0])
        self.image = self.resize_image(self.image, 800, 600)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(20, 20, anchor=tk.NW, image=self.tk_image)
        self.reference = self.tk_image
        

        # Draw the current crop region
        if self.cam_list!=None:
            w = self.cam_list[0]['width'] * self.image.width
            h = self.cam_list[0]['height'] * self.image.height
            x = self.cam_list[0]['x'] * self.image.width
            y = self.cam_list[0]['y'] * self.image.height

            self.rect = self.canvas.create_rectangle(x, y, x+w, y+h , outline="red")

        # Uncomment the following line if you want to accumulate regions for each image
        # values.append(np.copy(self.region))
        # self.submitted = False
        # return values
    
    def save_region(self):
        # Save the current region and proceed to the next image
        self.regions.append(self.region)
        self.submitted = False 

        if len(self.images) > 1:
            self.images = self.images[1:]
            if self.cam_list!=None:
                self.cam_list = self.cam_list[1:]
            self.getImageCropRegions(self.images)
        else:
            print("No more photos to crop.")
            self.root.destroy()
    
    def generate_regions(self):
        # Return all the saved regions
        return self.regions

# https://stackoverflow.com/questions/50398649/python-tkinter-tk-support-checklist-box
class ChecklistBox:
    def __init__(self, root, choices):
        self.root = root
        self.root.title('Recording List')

        header = tk.Label(root, text="Record From:", font=('Arial',9,'bold','underline'))
        header.pack(side='top',pady=5)
        self.terminated = False

        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)

        self.vars = []
        for choice in choices:
            var = StringVar(value=choice)
            self.vars.append(var)
            cb = Checkbutton(root, var=var, text=choice,
                                onvalue=choice, offvalue="", width=20, 
                                relief="flat", highlightthickness=0
            )
            cb.pack(side="top", fill="x",pady=5)

        self.crop_button = Button(root, text="Submit", command=self.root.destroy)
        self.crop_button.pack(pady=(20,30))


    def getCheckedItems(self):
        values = []
        for var in self.vars:
            value = var.get()
            if value:
                values.append(value)
        return values
    def close_gracefully(self):
        self.terminated = True
        self.root.destroy()
    
    
class RecordingDetails:
    

    def __init__(self, root, cam_names, model_names):
        self.root = root
        self.root.title('Recording Details')

        self.cam_names = cam_names
        self.model_names = model_names
        self.terminated = False
        
        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)

        
        self.content = tk.Frame(root)
        numcols = (len(model_names)+3)
        numrows = (len(cam_names)+3)

        self.settings = {}
        for i in cam_names:
            self.settings[i] = [{'Time': tk.StringVar(value=1)},{'Segment': tk.StringVar(value=30)}]
            for x in model_names:
                self.settings[i].append({x: tk.BooleanVar(value=False)})

        # force the number of columns to be odd
        if numcols%2==0:
            numcols += 1 

        #self.content.grid(column=0, row=0, columnspan=numcols, rowspan=numrows)
        self.content.grid_rowconfigure(numrows)
        self.content.grid_columnconfigure(numcols)
        # build column labels
        colLabels = [
            tk.Label(self.content, text='Camera', font=('Arial',9,'bold','underline')),
            tk.Label(self.content, text='Recording Time (days)', font=('Arial',9,'bold','underline')),
            tk.Label(self.content, text='Segment Length (mins)', font=('Arial',9,'bold','underline'))
        ]

        for i in model_names:
            colLabels.append(tk.Label(self.content, text=i, font=('Arial',9,'bold','underline')))

        # draw column labels
        for i, lbl in enumerate(colLabels):
            lbl.grid(column=i, row=0)

        # make the camera labels
        for i,cam in enumerate(cam_names):
            lb = tk.Label(self.content, text=cam)
            lb.grid(column=0, row=i+1)

        for x in range(1, len(cam_names) + 1):
            cam = cam_names[x - 1]  # Correctly assign the camera name

            te = tk.Entry(self.content, textvariable=self.settings[cam][0]['Time'])
            te.grid(column=1, row=x)
            se = tk.Entry(self.content, textvariable=self.settings[cam][1]['Segment'])
            se.grid(column=2, row=x)

            for y, name in enumerate(model_names):
                mdn = Checkbutton(self.content, var=self.settings[cam][y + 2][name], onvalue=True, offvalue=False)
                mdn.grid(column=y + 3, row=x)


        
        self.content.pack(pady=(20,30)) 

        self.submit_button = Button(root, text="Submit", command=self.kill_if_done)
        self.submit_button.pack(pady=(20,30))

    def validateVals(self, dict):
        for cam in dict.keys():
            settings_list = dict[cam]
            try:
                x = int(settings_list[0]['Time'])
                y = int(settings_list[1]['Segment'])
            except:
                return False
            if x>0 or y>0:
                return True
            return False

    def getVals(self):
        outputDict = {}

        for key in self.settings.keys():
            outputDict[key] = [{'Time': self.settings[key][0]['Time'].get()},{'Segment': self.settings[key][1]['Segment'].get()}]
            for i,x in enumerate(self.model_names):
                outputDict[key].append({x: self.settings[key][i+2][x].get()})

        if self.validateVals(outputDict):
            return outputDict
        return None
        
    def kill_if_done(self):
        if self.getVals()!=None:
            self.root.destroy()
    def close_gracefully(self):
        self.terminated = True
        self.root.destroy()
            
                


class ProcessMonitor:
    def __init__(self, root, pids, lookupCam, lookupProcess, recordingConfig):
        self.root = root
        self.root.title('Process Monitor')
        self.pids = pids 
        self.lookupCam = lookupCam
        self.lookupProcess = lookupProcess
        self.recordingConfig = recordingConfig
        self.content = None
        self.terminated = False
        
        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)

        self.update()

    
    def terminate(self, pid):
        try:
            process = self.lookupProcess[pid]
            process.stdin.write(b'q')
            process.stdin.flush()

            camera_killed = self.lookupCam[pid]

            # update the camera config
            with open(self.recordingConfig, 'r') as file:
                rconfig = yaml.safe_load(file)
            
            t = time.localtime()

            current_time = time.strftime("%H:%M:%S", t)

            rconfig['cameras_time'][camera_killed]['end_time'] = current_time

            with open(self.recordingConfig, 'w+') as file:
                yaml.dump(rconfig, file, allow_unicode=True)



        except Exception as e:
            print(f"Error sending quit command to ffmpeg: {e}")
    
    def update(self):

        if self.content!=None:
            self.content.destroy()
            self.close.destroy()
            self.warn.destroy()

        colors = []

        AllDead = True

        for pid in self.pids:
            alive = False
            if psutil.pid_exists(int(pid)):
                colors.append('green yellow')
                AllDead = False
                alive = True
            else:
                colors.append('firebrick1')

            if alive:
                # check for files ready to be inferenced
                print('checking for overflowing files')
            

        if AllDead:
            self.root.destroy()
            raise Exception('User terminated all processes.')
        
        self.content = tk.Frame(self.root)
        numcols = 3
        numrows = (len(self.pids)+2)
        self.content.grid_rowconfigure(numrows)
        self.content.grid_columnconfigure(numcols)

        cameras = tk.Label(self.content, text='Camera', font=('Arial',9,'bold','underline'))
        state = tk.Label(self.content, text="State", font=('Arial',9,'bold','underline'))

        cameras.grid(column=0,row=0,pady=5)
        state.grid(column=1,row=0,pady=5)
        

        for i,pid in enumerate(self.pids):
            cam = self.lookupCam[pid]
            name = tk.Label(self.content, text=cam)
            color = tk.Label(self.content, text="", bg=colors[i], width=3, borderwidth=3, relief="flat")
            terminate = tk.Button(self.content, text="Terminate", command=lambda: self.terminate(pid))

            name.grid(column=0,row=(i+2),pady=5)
            color.grid(column=1,row=(i+2),pady=5)
            terminate.grid(column=2,row=(i+2),pady=5)

        self.content.pack(side="top", pady=(10,5))

            
        self.warn = tk.Label(self.root, text="Warning, closing window will \nterminate the video acquisition process.", fg='tomato', padx=5)
        self.warn.pack(pady=(5,5))


        self.close = Button(self.root, text="TERMINATE ALL", command=self.root.destroy)
        self.close.pack(pady=(20,30))


        # wait some time and then check all the processes again
        self.root.after(2000, self.update)
    def close_gracefully(self):
        self.terminated = True
        self.root.destroy()



# Generate a single frame of a prerecorded video
def generate_image_prerecorded(video_in, frame_location):
    command = f"ffmpeg -i {video_in} -vf \"select=eq(n\,34)\" -vframes 1 -y {frame_location}"
    subprocess.call(command, shell=True)

# Generate a single frame of a stream
def generate_image(rtsp_url, frame_location):
    command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {frame_location}"
    subprocess.call(command, shell=True)

# Create an FFMPEG process for cropping the prerecorded video
def crop_video(video, cw, ch, cx, cy, output_loc):
    command = [
        'ffmpeg', '-i', str(video),
        '-filter_complex', f"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy})[cropped]", 
        '-map', '[cropped]', '-y', output_loc
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    return process

# Create an FFMPEG process for recording the stream
def record_stream(rtsp_url, name, video_dir, time, seg_time, cw, ch, cx, cy, scale, framerate):
    output_path = f"{video_dir}/recording_{name}_%05d.mp4"
    command = [
        'ffmpeg', '-rtsp_transport', 'tcp', '-i', str(rtsp_url), 
        '-r', str(framerate), '-t', str(time), 
        '-filter_complex', f"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale={scale}:{scale}[cropped]", 
        '-map', '[cropped]', '-f', 'segment', '-segment_time', str(seg_time), 
        '-reset_timestamps', '1', '-y', output_path
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    return process


def load_cameras(project_config='undefined'):
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

        return (cconfig, videos, frames, cameras)
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

        return (cconfig, videos, frames, cameras)

def load_recordings(project_config='undefined'):
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

        # grabbing the locations of the recording folder
        recordings = pconfig['recordings_path']

        return recordings
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
        recordings = pconfig['recordings_path']

        return recordings


def select_rois(project_config='undefined'):
    cconfig, videos, frames, cameras = load_cameras(project_config)
    processes = []
    images = []

    # grab the first frames for each camera in the camera list
    for i, cam in enumerate(cconfig['cameras']):
        frame_location = os.path.join(frames, cam['name']+'.jpg')
        process = Process(target=generate_image, args=(cam['rtsp_url'],frame_location))
        process.start()
        processes.append(process)
        images.append(frame_location)

    # wait for all processes to finish
    for process in processes:
        process.join()

    root = tk.Tk()
    root.geometry('840x600')
    app = ImageCropTool(root, images, cconfig)
    root.mainloop()

    if app.terminated:
        raise Exception('User terminated process.')
    
    # generate a 2d array of cams x regions
    values = app.generate_regions()

    # check to see if the number values equals the number of cameras
    num_cams = len(cconfig['cameras'])
    if len(values)!=num_cams:
        raise Exception('User aborted roi selection.')
    

    # update the roi values
    for i, cam in enumerate(cconfig['cameras']):
        
        cw = values[i][0]
        ch = values[i][1]
        cx = values[i][2]
        cy = values[i][3]

        cam['width'] = cw
        cam['height'] = ch 
        cam['x'] = cx
        cam['y'] = cy

    # update the camera config
    config = {
                    'contrast':cconfig['contrast'],
                    'brightness':cconfig['brightness'],
                    'framerate':cconfig['framerate'],
                    'scale':cconfig['scale'],
                    'cameras':cconfig['cameras']
                }

    with open(cameras, 'w+') as file:
        yaml.dump(config, file, allow_unicode=True)


def buildModelDict(model_names, values):
    modelDict = {key: [] for key in model_names}
    modelDict[None] = []
    


    for key, val in values.items():
        noneFlag = False
        for dict in val:
            for name in model_names:
                if name in dict:
                    if dict[name] is True:
                        noneFlag = True
                        modelDict[name].append(key)
        if not noneFlag:
            modelDict[None].append(key)


    return modelDict

def buildCameraDict(camera_names, values):
    cameraDict = {key: {'recording_length':0,'segment_length':0,'end_time':None} for key in camera_names}

    for key, val in values.items():
        time_len = val[0]['Time']
        seg_len = val[1]['Segment']
        cameraDict[key]['recording_length'] = time_len
        cameraDict[key]['segment_length'] = seg_len

    return cameraDict




class Watcher(FileSystemEventHandler):
    def __init__(self, recordingConfig):
        self.recordingConfig = recordingConfig
    def on_created(self, event):
        if event.is_directory:
            return
        print(f'New file created: {event.src_path}')

        pathname = event.src_path

        camera_name = os.path.split(os.path.split(pathname)[0])[1]
        # update the camera config
        with open(self.recordingConfig, 'r') as file:
            rconfig = yaml.safe_load(file)
    
        rconfig['cameras_files'][camera_name].append(os.path.split(pathname)[1])

        with open(self.recordingConfig, 'w+') as file:
            yaml.dump(rconfig, file, allow_unicode=True)




def record(project_config='undefined', safe=True):
    cconfig, videos, frames, cameras = load_cameras(project_config)

    # Create a simple pop-up to select which cameras to record from
    cams = [cam['name'] for cam in cconfig['cameras']]

    root = tk.Tk()
    app = ChecklistBox(root, cams)
    root.mainloop()
    
    if app.terminated:
        raise Exception('Cameras not selected, exiting.')

    selected = app.getCheckedItems()

    # Make a little pop-up to ask for the recording details
    model_names = []
    cam_names = selected 

    root = tk.Tk()
    app = RecordingDetails(root, cam_names, model_names)
    root.mainloop()

    if app.terminated:
        raise Exception('Recording details not entered, exiting.')

    # getting the settings dictionary
    settings = app.getVals()
    modelDict = buildModelDict(model_names, settings)
    cameraDict = buildCameraDict(cam_names, settings)

    # assume that the dictionary is {'model':['cam1','cam2']}, None is a valid model

    # if safe, let's call the select rois function, just so they know what they're getting and to see if we can access the cameras
    if safe:
        try:
            select_rois(project_config)
        except:
            raise Exception("Process terminated.")

    # Ok, let's go ahead and get those camera settings again
    cconfig, videos, frames, cameras = load_cameras(project_config)

    # limit this to only selected cams
    camera_hard_settings = [cam for cam in cconfig['cameras'] if cam['name'] in selected]

    # build out the recording folder
    recordings = load_recordings(project_config)
    recording_name = 'recording_'+datetime.utcnow().strftime('%Y%m%d%H%M%S')

    print(f"Opening a recording folder called {recording_name}")
    recording_folder = os.path.join(recordings, recording_name)
    
    # check for duplicates
    if os.path.exists(recording_folder):
        raise Exception('Somehow the recording name is a duplicate. Weird, try again.')
        

    os.mkdir(recording_folder)

    # Ok, let's make the recording config file
    config_file = os.path.join(recording_folder, 'details.yaml')

    config = {
        'start_date':'',
        'start_time':'',
        'cameras_per_model':{},
        'cameras_time':[],
        'cameras_files':[],
        'cameras':[]
    }

    config['cameras_per_model'] = modelDict
    config['cameras_time'] = cameraDict
    config['cameras'] = camera_hard_settings
    config['cameras_files'] = {c:[] for c in selected}

    cameras = camera_hard_settings

    # clear the old video files before starting
    for i, cam in enumerate(cameras):
        video_dir = cam['video_dir']
        files = glob.glob(os.path.join(video_dir, '*'))
        for file in files:
            if os.path.isfile(file):
                os.remove(file)

    t = time.localtime()

    # this is the superior date structure by the way
    current_date = time.strftime("%Y-%m-%d", t)

    current_time = time.strftime("%H:%M:%S", t)

    start_date = current_date
    start_time = current_time

    config['start_date'] = start_date
    config['start_time'] = start_time

    # start the recordings, if something fails, erase all of this scorched earth style
    processes = []
    watchdogs = []
    orchestrators = []

    for i, cam in enumerate(cameras):

        # record_stream(rtsp_url, name, video_dir, time, seg_time, cw, ch, cx, cy, scale, framerate)
        
        name = cam['name']
        framerate = cconfig['framerate']
        scale = cconfig['scale']
        cw = cam['width']
        ch = cam['height']
        cx = cam['x']
        cy = cam['y']
        rtsp_url = cam['rtsp_url']
        video_dir = cam['video_dir']

        recording_length = 1440*60 * int(config['cameras_time'][name]['recording_length'])
        segment_length = 60 * int(config['cameras_time'][name]['segment_length'])

        path = video_dir 
        event_handler = Watcher(config_file)
        observer = Observer()

        observer.schedule(event_handler, path, recursive=False)
        observer.start()

        process = record_stream(rtsp_url, name, video_dir, recording_length, segment_length, cw, ch, cx, cy, scale, framerate)
        processes.append((process, process.pid, name))
        

        watchdogs.append(observer)


    
    # assuming that the processes actually started and are working, dump the contents of the config file into the recording folder
    # careful, any fault here would destroy child processes
    try:
        with open(config_file, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)
    except:
        print('Failed to dump the camera settings.')

    # start up the models. This has the highest probablitity of breaking
    try:
        for key in modelDict.keys():
            # determine the type of orchestrator to make
            if key==None:
                process = Process(target=orchestrate.basic_orchestrator, args=(config_file,))
                process.start()
                orchestrators.append(process)


    except:

        raise Exception('Error loading a model. Aborting!')
    

    pids = [process[1] for process in processes]
    lookupCam = {process[1]:process[2] for process in processes}
    lookupProcess = {process[1]:process[0] for process in processes}

    root = tk.Tk()
    app = ProcessMonitor(root, pids, lookupCam, lookupProcess, config_file)
    root.mainloop()

    for pid in pids:
        app.terminate(pid)

    # we have reached the end of the recording, hopefully. time to clean up processes

    for wd in watchdogs:
        wd.stop()
        wd.join()
    

    for process in orchestrators:
        process.terminate()
        process.join()

def crop_prerecorded(video_location):

    processes = []
    images = []

    # Use glob to find all .mp4 files
    if not video_location.endswith('/'):
        video_location += '/'
    video_locations = glob.glob(video_location + '*.mp4')
    print(video_locations)

    # frame folder
    frame_folder = os.path.join(video_location, 'frames')
    if not os.path.isdir(frame_folder):
        os.mkdir(frame_folder)

    cropped_folder = os.path.join(video_location, 'cropped')
    if not os.path.isdir(cropped_folder):
        os.mkdir(cropped_folder)

    # grab the first frames for each video in the video list
    for i, vid in enumerate(video_locations):
        frame_location = os.path.join(frame_folder, os.path.splitext(os.path.split(vid)[1])[0]+".jpg")
        process = Process(target=generate_image_prerecorded, args=(vid,frame_location))
        process.start()
        processes.append(process)
        images.append(frame_location)

    # wait for all processes to finish
    for process in processes:
        process.join()

    root = tk.Tk()
    root.geometry('840x600')
    app = ImageCropTool(root, images)
    root.mainloop()

    if app.terminated:
        raise Exception('User terminated process.')
    
    # generate a 2d array of cams x regions
    values = app.generate_regions()
    processes = []

    for i, vid in enumerate(video_locations):
        
        cw = values[i][0]
        ch = values[i][1]
        cx = values[i][2]
        cy = values[i][3]

        # crop the videos
        vid_location = os.path.join(cropped_folder, os.path.splitext(os.path.split(vid)[1])[0]+"_cropped.mp4")
        process = crop_video(vid, cw, ch, cx, cy, vid_location) 
        processes.append(process)

    for process in processes:
        process.wait()






