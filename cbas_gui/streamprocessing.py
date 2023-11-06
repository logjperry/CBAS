import os


def frameGen(url,camName):
    
    sysCmd = f'ffmpeg -i {url} -frames:v 1 -y frames/{camName}.png' #builds command to generate the next screenshot
    os.system(sysCmd) #Grab frame from video file and save to frames folder
    return f'frames/{camName}.png' #returns directory of new screenshot

def cropVid(url, w, h, x, y, brightness, contrast, camName):
    #builds command based on parameters
    systemCmd = f'ffmpeg -i {url} -c copy -flags +global_header -f segment -segment_time 1800 -segment_format_options movflags=+faststart -reset_timestamps 1; -filter:v -y "crop={w}:{h}:{x}:{y}, eq=brightness={brightness}:contrast={contrast}" cropped_vid/{camName}%03d.mp4'
    os.system(systemCmd)
    
    