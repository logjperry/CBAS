import os

def mpegCapture(url):
    snapshotFiles = os.listdir('frames/') #Grabs list of all screen shots that currently exist
    snapshots = [x.removesuffix('.png') for x in snapshotFiles] #sanitizes the file extension of of the filenames
    snapshots.sort() #ensures list is sorted so the last element is the most recent SS
    # try catch block grabs the number of the last file in the list. If no file exist defaults to zero
    try:
        nextSS = snapshots[-1].removeprefix('cam')
    except:
        nextSS = 0
    nextSS = int(nextSS) + 1 #increments file number by 1

    sysCmd = f'ffmpeg -i {url} -frames:v 1 frames/cam{nextSS}.png' #builds command to generate the next screenshot
    os.system(sysCmd) #Grab frame from video file and save to frames folder
    return f'frames/cam{nextSS}.png' #returns directory of new screenshot
    