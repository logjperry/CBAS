import os


def frameGen(url, brightness, contrast, camName):
    #builds command to capture frame from live feed to display
    sysCmd = f'ffmpeg -rtsp_transport tcp -i {url} -frames:v 1 -vf "eq=brightness={brightness}:contrast={contrast}" -y frames/{camName}.png' #builds command to generate the next screenshot
    os.system(sysCmd) #Grab frame from video file and save to frames folder
    return f'frames/{camName}.png' #returns directory of new screenshot

def cropVid(url, w, h, x, y, brightness, contrast, camName):
    #builds command based on parameters.
    #slices live feed into 30min intervals, crops the video and adjusts the brightness and contrast based on user selected parameters
    #resizes the video to 256x256 
    systemCmd = f'ffmpeg -rtsp_transport tcp -i {url} -f segment -segment_time 1800 -segment_format_options movflags=+faststart -reset_timestamps 1 -vf "crop={w}:{h}:{x}:{y},scale=256:256,eq=brightness={brightness}:contrast={contrast}" cropped_vid/{camName}%03d.mp4'
    os.system(systemCmd)

#Testing video stream processing function   
#cropVid('rtsp://:8554/camera1',500,500,0,0,1,1,'cam1')