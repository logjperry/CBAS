import os

def cropVid(url, w, h, x, y, brightness, contrast):
    ls = url.split('/') #splits URL into list of strings using / as a delimiter
    filename = ls[-1] #Grabs the last element from the list which is the file name of the video. Extension is left on as it will be reused
    systemCmd = f'ffmpeg -i {url} -filter:v "crop={w}:{h}:{x}:{y}, eq=brightness={brightness}:contrast={contrast}" cropped_vid/cropped_{filename}'
    os.system(systemCmd)
    
