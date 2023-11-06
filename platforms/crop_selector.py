import tkinter as tk
from tkinter import filedialog, Button
from PIL import Image, ImageTk
import os
import subprocess
from multiprocessing import Process

class ImageCropTool:
    def __init__(self, root, images):
        self.root = root
        self.root.title('Image Crop Tool')

        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        menu = tk.Menu(self.root)
        self.root.config(menu=menu)
        file_menu = tk.Menu(menu)
        menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open...", command=self.open_image)

        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)

        self.crop_button = Button(root, text="Submit", command=self.crop_image)
        self.crop_button.pack(side=tk.BOTTOM)

        self.filename = None
        self.rect = None
        self.start_x = None
        self.start_y = None

        self.submitted = False 
        self.region = []
        self.regions = []

        self.images = images

        print(images)
        self.getImageCropRegions(images)

    def open_image(self):
        self.filename = filedialog.askopenfilename()
        if not self.filename:
            return

        self.image = Image.open(self.filename)
        self.image = self.resize_image(self.image, 800, 600)  # Resize to fit the window, adjust as needed
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def resize_image(self, image, width, height):
        img_aspect = image.width / image.height
        win_aspect = width / height

        if img_aspect > win_aspect:
            return image.resize((width, int(width/img_aspect)), Image.ANTIALIAS)
        else:
            return image.resize((int(height*img_aspect), height), Image.ANTIALIAS)

    def on_press(self, event):
        if not self.rect:
            self.start_x = event.x
            self.start_y = event.y
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline="red")

    def on_drag(self, event):
        self.canvas.coords(self.rect, self.start_x, self.start_y, event.x, event.y)

    def crop_image(self):
        print("here")
        if not self.rect:
            return

        x1, y1, x2, y2 = self.canvas.coords(self.rect)

        # Normalized values
        norm_width = (x2 - x1) / self.image.width
        norm_height = (y2 - y1) / self.image.height
        norm_x_origin = x1 / self.image.width
        norm_y_origin = y1 / self.image.height

        #output_format = f"width:{norm_width:.4f}:height:{norm_height:.4f}:x_origin:{norm_x_origin:.4f}:y_origin:{norm_y_origin:.4f}"
        #print(output_format)
        self.region = []

        self.region.append(norm_width)
        self.region.append(norm_height)
        self.region.append(norm_x_origin)
        self.region.append(norm_y_origin)

        self.submitted = True
        self.save_region()

        # Uncomment if you want to save the cropped image
        # cropped = self.image.crop((x1, y1, x2, y2))
        # cropped.save("cropped.png")
    
    def getImageCropRegions(self, im_list):

        values = []
        im = im_list[0]
        if hasattr(self, "image_on_canvas"):
            self.canvas.delete(self.image_on_canvas)
        
        # If there's an existing rectangle, remove it
        if self.rect:
            self.canvas.delete(self.rect)
            self.rect = None

        self.image = Image.open(im)
        self.image = self.resize_image(self.image, 800, 600)  # Resize to fit the window, adjust as needed
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.image_on_canvas = self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)


        #values.append(np.copy(self.region))
        #self.submitted = False
        #return values
    
    def save_region(self):
        self.regions.append(self.region)

        self.submitted = False 

        if len(self.images)>1:
            self.images = self.images[1:]
            self.getImageCropRegions(self.images)
        else:
            print("No more photos to crop.")
            self.root.destroy()
    
    def generate_regions(self):
        return self.regions
        

# list of RTSP IP addresses
#rtsp_ips = ["128.252.76.231", "128.252.128.251", "128.252.128.157", "128.252.76.205", "128.252.76.159", "128.252.76.221", "128.252.76.216", "128.252.76.171"]
rtsp_ips = ["192.168.1.53"]
#rtsp_ips = ["192.168.1.49", "192.168.1.50", "192.168.1.51", "192.168.1.52", "192.168.1.53", "192.168.1.54", "192.168.1.55", "192.168.1.56","192.168.1.57", "192.168.1.58", "192.168.1.59", "192.168.1.60"]
#rtsp_ips = ["192.168.1.53", "192.168.1.54"]
#rtsp_ips = ["192.168.1.53", "192.168.1.54"]
#rtsp_ips = ["192.168.1.53", "192.168.1.54"]

def generate_image(ip, cam_number, username, password):
    #ffmpeg -i rtsp://path/to/stream -vframes 1 output.jpg
    command = f"ffmpeg -rtsp_transport tcp -i rtsp://{username}:{password}@{ip}:8554/profile0 -vf \"select=eq(n\,34)\" -vframes 1 -y videos/cam{cam_number}/frame.jpg"
    subprocess.call(command, shell=True)

def record_stream(ip, cam_number, username, password, time, seg_time, cw, ch, cx, cy):
    command = f"ffmpeg -rtsp_transport tcp -i rtsp://{username}:{password}@{ip}:8554/profile0 -r 10 -t {time} -filter_complex \"[0:v]crop=(iw*{cw}):(ih*{ch}):(iw*{cx}):(ih*{cy}),scale=256:256[cropped]\" -map \"[cropped]\" -f segment -segment_time {seg_time} -reset_timestamps 1 -y videos/cam{cam_number}/recording_{cam_number}_%05d.mp4"
    subprocess.call(command, shell=True)

def main():
    processes = []

    print('Please enter the username of these cameras: ')
    username = input()
    print('Please enter the password of these cameras: ')
    password = input()

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
        process = Process(target=generate_image, args=(ip, i+1, username, password))
        process.start()
        processes.append(process)
        images.append(os.path.join(os.getcwd(),"videos/cam"+str(i+1)+"/frame.jpg"))

    # wait for all processes to finish
    for process in processes:
        process.join()
    
    print('here')

    root = tk.Tk()
    root.geometry('800x600')
    app = ImageCropTool(root, images)
    root.mainloop()
    
    
    values = app.generate_regions()

    
    print(values)
    

    # create and start a new process for each RTSP IP
    for i, ip in enumerate(rtsp_ips):
        # you can modify the filename pattern as per your needs
        cw = values[i][0]
        ch = values[i][1]
        cx = values[i][2]
        cy = values[i][3]
        for j in range(0,100):
            process = Process(target=record_stream, args=(ip, i+1, username, password, time, seg, cw, ch, cx, cy))
            process.start()
            processes.append(process)

    # wait for all processes to finish
    for process in processes:
        process.join()

if __name__ == "__main__":
    main()



