import tkinter as tk
from tkinter import filedialog, Button, StringVar, Checkbutton
from PIL import Image, ImageTk
import os
import subprocess
from multiprocessing import Process
import yaml
from sys import exit

class ImageCropTool:
    def __init__(self, root, images, cconfig):
        # Initialize the main window and canvas for the tool
        self.root = root
        self.root.title('Image Crop Tool')

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)


        # Bind mouse events for drawing the rectangle
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)

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
        self.cam_list = cconfig['cameras'].copy()

        # Process images for cropping
        self.getImageCropRegions(images)



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

        self.image = Image.open(im_list[0])
        self.image = self.resize_image(self.image, 800, 600)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(20, 20, anchor=tk.NW, image=self.tk_image)
        

        # Draw the current crop region
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

        self.vars = []
        for choice in choices:
            var = StringVar(value=choice)
            self.vars.append(var)
            cb = Checkbutton(root, var=var, text=choice,
                                onvalue=choice, offvalue="",
                                anchor="w", width=20, 
                                relief="flat", highlightthickness=0
            )
            cb.pack(side="top", fill="x", anchor="w")

        self.crop_button = Button(root, text="Submit", command=self.root.destroy)
        self.crop_button.pack(pady=(20,30))


    def getCheckedItems(self):
        values = []
        for var in self.vars:
            value = var.get()
            if value:
                values.append(value)
        return values
    
    
class RecordingDetails:
    

    def __init__(self, root, cam_names, model_names):
        self.root = root
        self.root.title('Recording Details')

        # adding an option for setting all of the cameras to the same values
        cam_names.append('all')
        
        self.content = tk.Frame(root)
        numcols = (len(model_names)+3)
        numrows = (len(cam_names)+3)

        self.settings = {}
        for i in cam_names:
            self.settings[i] = [{'Time': tk.StringVar(value='0')},{'Segment': tk.StringVar(value='0')}]
            for x in model_names:
                self.settings[i].append({x: tk.BooleanVar(value=True)})
        print(self.settings)

        # force the number of columns to be odd
        if numcols%2==0:
            numcols += 1 

        #self.content.grid(column=0, row=0, columnspan=numcols, rowspan=numrows)
        self.content.grid_rowconfigure(numrows)
        self.content.grid_columnconfigure(numcols)
        # build column labels
        colLabels = [
            tk.Label(self.content, text='Camera'),
            tk.Label(self.content, text='Recording Time (days)'),
            tk.Label(self.content, text='Segment Length (mins)')
        ]
        for i in model_names:
            colLabels.append(tk.Label(self.content, text=i))

        # draw column labels
        for i, lbl in enumerate(colLabels):
            lbl.grid(column=i, row=0)

        # make the camera labels
        for i,cam in enumerate(cam_names):
            lb = tk.Label(self.content, text=cam)
            lb.grid(column=0, row=i+1)

        for x in range(1, len(cam_names)+1):
            if x!=len(cam_names)+1:
                cam = cam_names[x-1]
            cam = 'all'
            
            te = tk.Entry(self.content)
            te.grid(column=1, row=x)
            se = tk.Entry(self.content)
            se.grid(column=2, row=x)


            for y, name in enumerate(model_names):
                mdn = Checkbutton(self.content, var=self.settings[cam][name], onvalue=True, offvalue=False)
                mdn.grid(column=y+3, row=x)


        self.content.pack(pady=(20,30)) 

        self.submit_button = Button(root, text="Submit", command=self.root.destroy)   
        self.submit_button.pack(pady=(20,30))    


# Generate a single frame of a stream
def generate_image(rtsp_url, frame_location):
    command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -vf \"select=eq(n\,34)\" -vframes 1 -y {frame_location}"
    subprocess.call(command, shell=True)

# Create an FFMPEG process for recording the stream
def record_stream(rtsp_url, name, video_dir, time, seg_time, cw, ch, cx, cy):
    command = f"ffmpeg -rtsp_transport tcp -i {rtsp_url} -r 10 -t {time} -filter_complex \"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale=256:256[cropped]\" -map \"[cropped]\" -f segment -segment_time {seg_time} -reset_timestamps 1 -y {video_dir}/recording_{name}_%05d.mp4"
    subprocess.call(command, shell=True)


def load_cameras(project_config='undefined'):
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

        return (cconfig, videos, frames, cameras)
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

        return (cconfig, videos, frames, cameras)


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
    
    # generate a 2d array of cams x regions
    values = app.generate_regions()

    # check to see if the number values equals the number of cameras
    num_cams = len(cconfig['cameras'])
    if len(values)!=num_cams:
        print('User aborted roi selection.')
        exit(0)
    

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

def monitor_processes(processes):
    # wait for all processes to finish
    for process in processes:
        process.join()



def record(project_config='undefined'):
    cconfig, videos, frames, cameras = load_cameras(project_config)

    # Create a simple pop-up to select which cameras to record from
    cams = [cam['name'] for cam in cconfig['cameras']]

    cams = ['cam1','cam2']

    root = tk.Tk()
    app = ChecklistBox(root, cams)
    root.mainloop()

    selected = app.getCheckedItems()

    # go ahead and get the real cameras
    cams = [cam for cam in cconfig['cameras'] if cam['name'] in selected]

    # Make a little pop-up to ask for the recording details
    model_names = ['model 1', 'model 2']
    cam_names = selected 

    root = tk.Tk()
    app = RecordingDetails(root, cam_names, model_names)
    root.mainloop()

    # make the recording record 



def main():
    processes = []

    print('Please enter the total amount of time of the recording (in days): ')
    time = float(input())
    time *= 1440*60

    print('Please enter the segmentation time of the recordings (in minutes): ')
    seg = int(input())
    seg *= 60

    images = []

    # create and start a new process for each RTSP IP
    for i, ip in enumerate(rtsp_ips):
        # you can modify the filename pattern as per your needs
        process = Process(target=generate_image, args=(ip, i+1, 'username', 'password'))
        process.start()
        processes.append(process)
        images.append(os.path.join(os.getcwd(),"videos/cam"+str(i+1)+"/frame.jpg"))

    # wait for all processes to finish
    for process in processes:
        process.join()

    root = tk.Tk()
    root.geometry('800x600')
    app = ImageCropTool(root, images)
    root.mainloop()
    
    
    values = app.generate_regions()
    

    # create and start a new process for each RTSP IP
    for i, ip in enumerate(rtsp_ips):
        # you can modify the filename pattern as per your needs
        cw = values[i][0]
        ch = values[i][1]
        cx = values[i][2]
        cy = values[i][3]
        process = Process(target=record_stream, args=(ip, i+1, 'username', 'password', time, seg, cw, ch, cx, cy))
        process.start()
        processes.append(process)

    # wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()



