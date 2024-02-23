
import os
import sys 
import yaml
import time

def inference(config_path, name, video_path, storage_path):
    import deeplabcut

    if video_path!='' and storage_path!='':

        lock_file = None
        while True:

            ready = []
            finished = []
            wrapup = False


            
            if os.path.exists(video_path):
                with open(video_path, 'r') as file:
                    iconfig = yaml.safe_load(file)
                    finished = iconfig['finished']
                    ready = iconfig['ready']
                    wrapup = iconfig['wrapup']
                    lock_file = iconfig['lock']
            else:
                time.sleep(5)
                continue   


            with open(lock_file, 'r') as file:
                lconfig = yaml.safe_load(file)
            lock = lconfig['lock']
            
            if lock != name:
                done = False
                while not done:
                    time.sleep(5)
                    with open(lock_file, 'r') as file:
                        lconfig = yaml.safe_load(file)
                    lock = lconfig['lock']
            
                    if lock == name:
                        done = True

                            

            # start the timer
            start_time = time.time()
            
            with open(video_path, 'r') as file:
                iconfig = yaml.safe_load(file)
                finished = iconfig['finished']
                ready = iconfig['ready']
                wrapup = iconfig['wrapup']
                lock_file = iconfig['lock']
                times = iconfig['times']
                avg_times = iconfig['avg_times']

            videos_to_add = ready

            if len(videos_to_add)==0 and wrapup:
                break
            
            if len(videos_to_add) != 0:
                success = False

                if not success:
                    good_videos = []
                    for vid in videos_to_add:
                        sub_list = [vid]
                        temp_storage_path = os.path.join(storage_path, os.path.splitext(os.path.split(vid)[1])[0])
                        good_videos.append(temp_storage_path)
                        try:
                            deeplabcut.analyze_videos(config_path, sub_list, shuffle=1, save_as_csv=True, videotype='.mp4', destfolder=temp_storage_path)
                        except:
                            print(f'There is a problem with infering predictions for this video, {vid}.')
                    
                    videos_to_add = good_videos


            for vid in videos_to_add:
                if vid not in finished:
                    finished.append(os.path.splitext(vid)[0])

            # calculate the total time
            total_time = time.time() - start_time
            avg_time = 0
        
            if len(videos_to_add)>0:
                avg_time = total_time/len(videos_to_add)

            
            with open(video_path, 'r') as file:
                iconfig = yaml.safe_load(file)
                wrapup = iconfig['wrapup']
            
            times.append(total_time)
            avg_times.append(avg_time)
                
            iconfig = {'ready':[],'finished':finished,'wrapup':wrapup,'lock':lock_file, 'avg_times':avg_times, 'times':times, 'iter':iconfig['iter']+1}

                
            with open(video_path, 'w') as file:
                yaml.dump(iconfig, file, allow_unicode=True)

            with open(lock_file, 'r') as file:
                lconfig = yaml.safe_load(file)
            
            lconfig = {'lock':None,'dead':lconfig['dead']}
            with open(lock_file, 'w') as file:
                yaml.dump(lconfig, file, allow_unicode=True)
        
        
        with open(lock_file, 'r') as file:
            lconfig = yaml.safe_load(file)

        lconfig['dead'].append(name)

        lconfig = {'lock':lconfig['lock'],'dead':lconfig['dead']}
        with open(lock_file, 'w') as file:
            yaml.dump(lconfig, file, allow_unicode=True)

if __name__ == '__main__':
    inference(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])