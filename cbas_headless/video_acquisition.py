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
from cbas_headless import orchestrate
import ttkbootstrap as ttk
from ttkbootstrap.constants import *

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time


theme = 'superhero'

# A basic GUI for selecting the region of interest for each camera
class ImageCropTool:
    def __init__(self, root, images, cconfig=None, selected=None):
        # Initialize the main window and canvas for the tool
        self.root = root
        self.root.title('Image Crop Tool')

        self.canvas = tk.Canvas(self.root, bg='grey')
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.terminated = False


        # Bind mouse events for drawing the rectangle
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)

        # Button for saving the region of interest (ROI)
        self.crop_button = Button(root, text="Save ROI", command=self.crop_image, font=('TkDefaultFixed', 15), relief='flat', autostyle=False)
        self.crop_button.pack(pady=(5,5))

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
            
            if selected is not None:
                self.cam_list = []
                for cam in cconfig['cameras']:
                    if cam['name'] in selected:
                        self.cam_list.append(cam.copy())
                
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
            new_width = width
            new_height = int(width / img_aspect)
        else:
            new_width = int(height * img_aspect)
            new_height = height

        

        return image.resize((new_width, new_height), Image.ANTIALIAS)

    def on_press(self, event):
        # Start drawing the rectangle from where the mouse is clicked
        self.canvas.delete(self.rect)
        self.start_x = event.x
        self.start_y = event.y
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="black", width=5)
        

    def on_drag(self, event):
        # Adjust the rectangle dimensions as the mouse is dragged
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def crop_image(self):
        # Function to save the cropped region
        if not self.rect:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.rect)

        x1 = x1-5
        y1 = y1-5
        x2 = x2-5
        y2 = y2-5

        # Calculate normalized values for the crop region
        norm_width = (x2 - x1) / self.image.width
        norm_height = (y2 - y1) / self.image.height
        norm_x_origin = x1 / self.image.width
        norm_y_origin = y1 / self.image.height

        self.region = [norm_width, norm_height, norm_x_origin, norm_y_origin]

        self.submitted = True
        self.save_region()
    
    def getImageCropRegions(self, im_list):
        # Function to handle multiple images for cropping
        if hasattr(self, "image_on_canvas"):
            self.canvas.delete(self.image_on_canvas)
        
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        if not os.path.exists(im_list[0]):
            raise Exception('Error loading frame image.')

        self.image = Image.open(im_list[0])
        self.image = self.resize_image(self.image, 800, 600)

        self.canvas.config(width=self.image.width+10, height=self.image.height+10)

        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(5, 5, anchor=tk.NW, image=self.tk_image)
        self.reference = self.tk_image
        

        # Draw the current crop region
        if self.cam_list!=None:
            w = self.cam_list[0]['width'] * self.image.width
            h = self.cam_list[0]['height'] * self.image.height
            x = self.cam_list[0]['x'] * self.image.width + 5
            y = self.cam_list[0]['y'] * self.image.height + 5

            self.rect = self.canvas.create_rectangle(x, y, x+w, y+h , outline="black", width=5)

    
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

# A basic GUI for selecting the cameras to record from
class CameraChecklist:
    def __init__(self, root, choices):
        self.root = root
        self.root.title('Recording List')

        header = tk.Label(root, text="Record From:", font=('TkDefaultFixed', 20))
        header.grid(row=0, column=0,pady=5)

        middle = tk.Frame(root)
        middle.grid(row=1, column=0, pady=5)

        self.terminated = False

        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)

        max_colnum = 5

        self.vars = []
        ind = 0
        for choice in choices:
            var = StringVar(value=choice)
            self.vars.append(var)
            cb = Checkbutton(middle, var=var, text=choice,
                                onvalue=choice, offvalue="", width=20, 
                                relief="flat", highlightthickness=0, font=('TkDefaultFixed', 10)
            )
            cb.deselect()
            cb.grid(row=ind%max_colnum, column=int(ind/max_colnum),pady=5)
            ind+=1

        self.crop_button = Button(root, text="Submit", font=('TkDefaultFixed', 15), relief='flat', command=self.root.destroy, autostyle=False)
        self.crop_button.grid(row=2, column=0, pady=10)


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

# A basic GUI for selecting the inference details of each camera 
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
            self.settings[i] = [{'Segment': tk.StringVar(value=30)}]
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
            tk.Label(self.content, text='Camera', font=('TkDefaultFixed', 15, 'underline'), padx=5),
            tk.Label(self.content, text='Segment Length (mins)', font=('TkDefaultFixed', 15,'underline'), padx=5)
        ]

        for i in model_names:
            colLabels.append(tk.Label(self.content, text=i, font=('TkDefaultFixed', 15, 'underline'), padx=5))

        # draw column labels
        for i, lbl in enumerate(colLabels):
            lbl.grid(column=i, row=0)

        # make the camera labels
        for i,cam in enumerate(cam_names):
            lb = tk.Label(self.content, text=cam, font=('TkDefaultFixed', 10))
            lb.grid(column=0, row=i+1)

        for x in range(1, len(cam_names) + 1):
            cam = cam_names[x - 1]  # Correctly assign the camera name
            se = tk.Entry(self.content, textvariable=self.settings[cam][0]['Segment'], font=('TkDefaultFixed', 10))
            se.grid(column=1, row=x)

            for y, name in enumerate(model_names):
                mdn = Checkbutton(self.content, var=self.settings[cam][y + 1][name], onvalue=True, offvalue=False, font=('TkDefaultFixed', 10))
                mdn.grid(column=y + 2, row=x)


        
        self.content.pack(pady=(10,10)) 

        self.submit_button = Button(root, text="Submit", font=('TkDefaultFixed', 15), command=self.kill_if_done, relief='flat', autostyle=False)
        self.submit_button.pack(pady=(10,10))

    def validateVals(self, dict):
        for cam in dict.keys():
            settings_list = dict[cam]
            try:
                y = int(settings_list[0]['Segment'])
            except:
                return False
            if y>0:
                return True
            return False

    def getVals(self):
        outputDict = {}

        for key in self.settings.keys():
            outputDict[key] = [{'Segment': self.settings[key][0]['Segment'].get()}]
            for i,x in enumerate(self.model_names):
                outputDict[key].append({x: self.settings[key][i+1][x].get()})

        if self.validateVals(outputDict):
            return outputDict
        return None
        
    def kill_if_done(self):
        if self.getVals()!=None:
            self.root.destroy()

    def close_gracefully(self):
        self.terminated = True
        self.root.destroy()

# A monitor for watching the recording and inferencing in real-time          
class ProcessMonitor:
    def __init__(self, root, pids, lookupCam, lookupProcess, recordingConfig, frames, modelcfgs=[], lockcfg=None):
        self.root = root
        self.root.title('Process Monitor')
        self.pids = pids 
        self.lookupCam = lookupCam
        self.lookupProcess = lookupProcess
        self.recordingConfig = recordingConfig
        self.content = None
        self.terminated = False
        self.modelcfgs = modelcfgs
        self.lockcfg = lockcfg
        
        # Exit gracefully
        root.protocol("WM_DELETE_WINDOW", self.close_gracefully)


        self.container = tk.Frame(self.root)

        self.content = tk.Frame(self.container)
        
        self.topcontent = tk.Frame(self.content)
        self.botcontent = tk.Frame(self.content)

        self.canvas = tk.Canvas(self.container, width=600, height=600, bg='grey')

        self.frames = frames

        # grab the top pid to draw first
        img_pid = pids[0]
        cam = lookupCam[img_pid]
        frame_location = os.path.join(frames, cam['name']+'.jpg')
        generate_image(cam['rtsp_url'],frame_location)

        self.image_path = frame_location

        self.draw_image()

        self.init_cameras(self.topcontent)
        self.init_models(self.botcontent)

        self.content.pack(side="left", pady=(10,5), padx=(10,10))
        
        self.canvas.pack(side="left", pady=(10,5), padx=(10,10))

        self.container.pack(side='top')

            
        self.warn = tk.Label(self.root, text="Warning, closing window will \nterminate the video acquisition process.", fg='tomato', padx=5)
        self.warn.pack(pady=(5,5))


        self.close = tk.Button(self.root, text="TERMINATE RECORDING", command=self.root.destroy, bg='firebrick3', font=('TkDefaultFixed', 15), relief='flat', autostyle=False)
        self.close.pack(pady=(10,10))

        self.update()
    
    def init_cameras(self, topcontent):
        numcols = 3
        numrows = (len(self.pids)+2)
        self.topcontent.grid_rowconfigure(numrows)
        self.topcontent.grid_columnconfigure(numcols)

        self.cameras = tk.Label(self.topcontent, text='Camera', font=('TkDefaultFixed', 15,'underline'))

        self.cameras.grid(column=0,row=0,pady=5)
        

        self.colors = {pid:None for pid in self.pids}

        colors = []

        AllDead = True

        for pid in self.pids:
            alive = False
            if psutil.pid_exists(int(pid)):
                colors.append('dark green')
                AllDead = False
                alive = True
            else:
                colors.append('firebrick3')
        
        if AllDead:
            self.root.destroy()
            raise Exception('User terminated all processes.')

        for i,pid in enumerate(self.pids):
            cam = self.lookupCam[pid]['name']
            name = tk.Label(self.topcontent, text=cam, font=('TkDefaultFixed', 10))
            status = ''
            if colors[i] == 'dark green':
                status = 'active'
            else:
                status = 'dead'
            self.colors[pid] = tk.Label(self.topcontent, text=status, bg=colors[i], relief='flat', borderwidth=4, font=('TkDefaultFixed', 10))
            disp = tk.Button(self.topcontent, text="Display", command=lambda id = pid: self.new_image(id), font=('TkDefaultFixed', 10), relief='flat', bg='grey', autostyle=False)
            terminate = tk.Button(self.topcontent, text="Terminate", command=lambda id = pid: self.terminate(id), bg='firebrick3', font=('TkDefaultFixed', 10), relief='flat', autostyle=False)

            name.grid(column=0,row=(i+2),pady=5)
            self.colors[pid].grid(column=1,row=(i+2),pady=5,padx=5)
            disp.grid(column=2,row=(i+2),pady=5,padx=5)
            terminate.grid(column=3,row=(i+2),pady=5,padx=5)

        self.topcontent.pack(anchor = 'nw', side="top", pady=(10,5), padx=(10,10))

    def init_models(self, botcontent):

        if len(self.modelcfgs)!=0:
            numcols = 2
            numrows = (len(self.modelcfgs)+2)
            self.botcontent.grid_rowconfigure(numrows)
            self.botcontent.grid_columnconfigure(numcols)

            self.models = tk.Label(self.botcontent, text='Model', font=('TkDefaultFixed', 15,'underline'))

            self.models.grid(column=0,row=0,pady=5)

            self.colorsM = {os.path.splitext(os.path.split(m)[1])[0]:None for m in self.modelcfgs}

            colors = []

            
            with open(self.lockcfg, 'r') as file:
                lconfig = yaml.safe_load(file)

            for m in self.modelcfgs:
                name = os.path.splitext(os.path.split(m)[1])[0]
                if lconfig['lock'] == name:
                    colors.append('dark green')
                else:
                    colors.append('blue')

            for i,m in enumerate(self.modelcfgs):
                name = os.path.splitext(os.path.split(m)[1])[0]
                n = tk.Label(self.botcontent, text=name, font=('TkDefaultFixed', 10))

                status = ''
                if colors[i] == 'blue':
                    status = 'paused'
                else:
                    status = 'active'
                self.colorsM[name] = tk.Label(self.botcontent, text=status, bg=colors[i], relief='flat', borderwidth=4, font=('TkDefaultFixed', 10))

                n.grid(column=0,row=(i+2),pady=5)
                self.colorsM[name].grid(column=1,row=(i+2),pady=5,padx=5)

            self.botcontent.pack(anchor = 'nw', side="bottom", pady=(10,5), padx=(10,10))


    def new_image(self, pid):
        img_pid = pid
        cam = self.lookupCam[img_pid]
        frame_location = os.path.join(self.frames, cam['name']+'.jpg')
        generate_image(cam['rtsp_url'],frame_location)

        self.image_path = frame_location

        self.draw_image()
    
    def terminate(self, pid):

        attempts = 100

        try:
            process = self.lookupProcess[pid]
            process.stdin.write(b'q')
            process.stdin.flush()

            time.sleep(1)
            process.kill()

            camera_killed = self.lookupCam[pid]['name']

            # update the camera config
            with open(self.recordingConfig, 'r') as file:
                rconfig = yaml.safe_load(file)
            
            t = time.localtime()

            current_time = time.strftime("%H:%M:%S", t)


            success = False 
            trys = 0
            while not success:
                try:
                    rconfig['cameras_time'][camera_killed]['end_time'] = current_time

                    with open(self.recordingConfig, 'w+') as file:
                        yaml.dump(rconfig, file, allow_unicode=True)
                    success = True
                except Exception as e:
                    success = False
                    time.sleep(1)
                    trys+=1

                    if trys>attempts:
                        print(e.message, e.args)




        except Exception as e:

            camera_killed = self.lookupCam[pid]['name']

            # update the camera config
            with open(self.recordingConfig, 'r') as file:
                rconfig = yaml.safe_load(file)
            
            t = time.localtime()

            current_time = time.strftime("%H:%M:%S", t)

            rconfig['cameras_time'][camera_killed]['end_time'] = current_time

            with open(self.recordingConfig, 'w+') as file:
                yaml.dump(rconfig, file, allow_unicode=True)

            print(f"Error sending quit command to ffmpeg: {e}")

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

    def draw_image(self):

        self.image = Image.open(self.image_path)
        self.image = self.resize_image(self.image, 600, 600)
        self.canvas.config(width=self.image.width+10,height=self.image.height+10)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(5, 5, anchor=tk.NW, image=self.tk_image)

        self.reference = self.tk_image

    def update(self):

        colors = []

        AllDead = True

        for pid in self.pids:
            alive = False
            if psutil.pid_exists(int(pid)):
                colors.append('dark green')
                AllDead = False
                alive = True
            else:
                colors.append('firebrick3')

        for i,pid in enumerate(self.pids):
            
            status = ''
            if colors[i] == 'dark green':
                status = 'active'
            else:
                status = 'dead'
                
            self.colors[pid].config(text=status, bg=colors[i])

            

        if AllDead:
            self.root.destroy()
            raise Exception('User terminated all processes.')


        if len(self.modelcfgs)!=0:
            colors = []

                
            with open(self.lockcfg, 'r') as file:
                lconfig = yaml.safe_load(file)

            for m in self.modelcfgs:
                name = os.path.splitext(os.path.split(m)[1])[0]
                if lconfig['lock'] == name:
                    colors.append('dark green')
                else:
                    colors.append('blue')

            for i,m in enumerate(self.modelcfgs):
                name = os.path.splitext(os.path.split(m)[1])[0]
                status = ''
                if colors[i] == 'dark green':
                    status = 'active'
                else:
                    status = 'paused'
                    
                self.colorsM[name].config(text=status,bg=colors[i])
        


        # wait some time and then check all the processes again
        self.root.after(2000, self.update)
    def close_gracefully(self):

        for pid in self.pids:
            self.terminate(pid)

        self.terminated = True
        self.root.destroy()


# Returns the models being used and the cameras that correspond to each model
def buildModelDict(model_names, values):
    modelDict = {key: [] for key in model_names}
    

    for key, val in values.items():
        for dict in val:
            for name in model_names:
                if name in dict:
                    if dict[name] is True:
                        modelDict[name].append(key)


    return modelDict

# Returns the camera settings being used
def buildCameraDict(camera_names, values):

    cameraDict = {key: {'segment_length':0,'end_time':None} for key in camera_names}

    for key, val in values.items():
        seg_len = val[0]['Segment']
        cameraDict[key]['segment_length'] = seg_len

    return cameraDict


# Generate a single frame of a prerecorded video
def generate_image_prerecorded(video_in, frame_location):
    command = f"ffmpeg -loglevel panic -i {video_in} -vf \"select=eq(n\,34)\" -vframes 1 -y {frame_location}"
    subprocess.call(command, shell=True)

# Generate cropped versions of prerecorded videos
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

# Generate a single frame of a stream
def generate_image(rtsp_url, frame_location):
    command = f"ffmpeg -loglevel panic  -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {frame_location}"
    subprocess.call(command, shell=True)

# Create an FFMPEG process for cropping the prerecorded video
def crop_video(video, cw, ch, cx, cy, output_loc):
    command = [
        'ffmpeg', '-loglevel', 'panic', '-i', str(video),
        '-filter_complex', f"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy})[cropped]", 
        '-map', '[cropped]', '-y', output_loc
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    return process

# Create an FFMPEG process for recording the stream
def record_stream(rtsp_url, name, video_dir, seg_time, cw, ch, cx, cy, scale, framerate):
    output_path = f"{video_dir}/recording_{name}_%05d.mp4"
    command = [
        'ffmpeg', '-loglevel', 'panic', '-rtsp_transport', 'tcp', '-i', str(rtsp_url), 
        '-r', str(framerate), 
        '-filter_complex', f"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale={scale}:{scale}[cropped]", 
        '-map', '[cropped]', '-f', 'segment', '-segment_time', str(seg_time), 
        '-reset_timestamps', '1', '-y', output_path
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    return process

# Gets the camera configurations from the project
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

# Gets the model configurations from the project
def load_models(project_config='undefined'):
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

        # grabbing the locations of the models
        models = pconfig['models_path']
        # extract the mdoels config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the models config file. Check for yaml syntax errors.')

        return mconfig
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

        # grabbing the locations of the models
        models = pconfig['models_path']
        # extract the mdoels config file
        try:
            with open(models, 'r') as file:
                mconfig = yaml.safe_load(file)
        except:
            raise Exception('Failed to extract the contents of the models config file. Check for yaml syntax errors.')

        return mconfig

# Gets the recording configurations from the project
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


#### User Functions ####

# Allows the user to set the ROIs for the cameras
def select_rois(project_config='undefined', selected=None):

    """
    A simple GUI for selecting the ROIs for the cameras. This function will prompt the user to select the ROIs for the cameras.

    Parameters
    ----------
    project_config : str, optional
        The path to the project_config.yaml file. If not provided, the function will assume the user is located within an active project. The default is 'undefined'.
    selected : list, optional
        A list of camera names that the user wants to select the ROIs for. If not provided, the function will assume the user wants to select the ROIs for all cameras. The default is None.

    """

    cconfig, videos, frames, cameras = load_cameras(project_config)
    processes = []
    images = []

    # grab the first frames for each camera in the camera list
    indices = []
    for i, cam in enumerate(cconfig['cameras']):
        if selected is not None and cam['name'] in selected:
            indices.append(i)
            frame_location = os.path.join(frames, cam['name']+'.jpg')
            process = Process(target=generate_image, args=(cam['rtsp_url'],frame_location))
            process.start()
            processes.append(process)
            images.append(frame_location)
        elif selected is None:
            indices.append(i)
            frame_location = os.path.join(frames, cam['name']+'.jpg')
            process = Process(target=generate_image, args=(cam['rtsp_url'],frame_location))
            process.start()
            processes.append(process)
            images.append(frame_location)

    # wait for all processes to finish
    for process in processes:
        process.join()

    root = ttk.Window(themename=theme)
    app = ImageCropTool(root, images, cconfig, selected=selected)
    root.mainloop()

    if app.terminated:
        raise Exception('User terminated process.')
    
    # generate a 2d array of cams x regions
    values = app.generate_regions()

    # check to see if the number values equals the number of cameras
    num_cams = len(selected)
    if len(values)!=num_cams:
        raise Exception('User aborted roi selection.')
    

    # update the roi values
    for i, cam in enumerate(cconfig['cameras']):
        ind = i
        if i not in indices:
            continue 
        else:
            ind = indices.index(i)

        if selected is not None and cam['name'] in selected:
        
            cw = values[ind][0]
            ch = values[ind][1]
            cx = values[ind][2]
            cy = values[ind][3]

            cam['width'] = cw
            cam['height'] = ch 
            cam['x'] = cx
            cam['y'] = cy
        elif selected is None:
            cw = values[ind][0]
            ch = values[ind][1]
            cx = values[ind][2]
            cy = values[ind][3]

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

# Records, infers, and stores videos. This is where the magic happens!
def record(project_config='undefined', safe=True):


    """
    The main function for recording videos from the cameras. This function will also handle the inference and storage of the videos.
    
    Parameters
    ----------
    project_config : str, optional
        The path to the project_config.yaml file. If not provided, the function will assume the user is located within an active project. The default is 'undefined'.
    safe : bool, optional
        If True, the function will prompt the user to select the ROIs for the cameras. The default is True.

    """

    cconfig, videos, frames, cameras = load_cameras(project_config)

    # the failure threshold, try to keep the ball rolling for x attempts, for non-critical things give much fewer attempts
    attempts = 100
    not_critical_attempts = 5

    # Create a simple pop-up to select which cameras to record from
    cams = [cam['name'] for cam in cconfig['cameras']]

    cams.sort()
    cams.sort(key=len)

    root = ttk.Window(themename=theme)
    app = CameraChecklist(root, cams)
    root.mainloop()
    
    if app.terminated:
        raise Exception('Cameras not selected, exiting.')

    selected = app.getCheckedItems()

    # Make a little pop-up to ask for the recording details
    model_names = []

    # Load the models
    models = load_models(project_config)['models']
    model_names = [m['name'] for m in models]

    cam_names = selected 

    root = ttk.Window(themename=theme)
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
            select_rois(project_config, selected=cam_names)
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
        'scale':cconfig['scale'],
        'framerate':cconfig['framerate'],
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
    config['cameras_files'] = {c:{} for c in selected}

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

    # start the recordings
    processes = []
    orchestrators = []

    for i, cam in enumerate(cameras):

        name = cam['name']
        framerate = cconfig['framerate']
        scale = cconfig['scale']
        cw = cam['width']
        ch = cam['height']
        cx = cam['x']
        cy = cam['y']
        rtsp_url = cam['rtsp_url']
        video_dir = cam['video_dir']

        segment_length = 60 * int(config['cameras_time'][name]['segment_length'])

        path = video_dir 

        process = record_stream(rtsp_url, name, video_dir, segment_length, cw, ch, cx, cy, scale, framerate)
        processes.append((process, process.pid, cam))
        

    # assuming that the processes actually started and are working, dump the contents of the config file into the recording folder
    # careful, any fault here would be very bad
    try:
        with open(config_file, 'w+') as file:
            yaml.dump(config, file, allow_unicode=True)
    except:
        print('Failed to dump the camera settings.')

    # start up the orchestrators. This has the highest probability of ending badly
    storage = None
    inference = None
    try:
        storage = Process(target=orchestrate.storage_orchestrator, args=(config_file,))
        storage.start()

        models_used = []

        for key in modelDict.keys():
            
            if len(modelDict[key]) > 0:
                models_used.append(key)

        if len(models_used) > 0:
            inference = Process(target=orchestrate.inference_orchestrator, args=(config_file, models,))
            inference.start()

    except:

        raise Exception('Error loading a model. Aborting!')
    

    pids = [process[1] for process in processes]
    lookupCam = {process[1]:process[2] for process in processes}
    lookupProcess = {process[1]:process[0] for process in processes}

    modelcfgs = []
    for m in models_used:
        path = os.path.join(os.path.split(config_file)[0], 'videos',m+'.yaml')
        modelcfgs.append(path)

    lockpath = None 
    if inference!=None:
        lockpath = os.path.join(os.path.split(config_file)[0], 'videos','lock.yaml')

    root = ttk.Window(themename=theme)
    app = ProcessMonitor(root, pids, lookupCam, lookupProcess, config_file, frames, modelcfgs=modelcfgs, lockcfg=lockpath)
    root.mainloop()

    # kill each of the video streams
    for pid in pids:
        app.terminate(pid)

    
    # kill the storage thread
    time.sleep(30)
    storage.terminate()
    storage.join()


    # wrap up the inferencing processes
    for m in models_used:
        
        path = os.path.join(os.path.split(config_file)[0], 'videos',m+'.yaml')

        success = False 
        trys = 0
        while not success:
            try:
                if os.path.exists(path):
                    with open(path, 'r') as file:
                        config = yaml.safe_load(file)
                        config['wrapup'] = True 
                    with open(path, 'w') as file:
                        yaml.dump(config, file, allow_unicode=True)
                success = True
            except Exception as e:
                success = False
                time.sleep(1)
                trys+=1

                if trys>attempts:
                    print(e.message, e.args)
        
    
    
    if inference!=None:
        inference.join()




