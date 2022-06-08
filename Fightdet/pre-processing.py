import os
import numpy as np
from Fightdet.video_to_numpy import video_to_numpy
from Fightdet.down_sample import down_sampling
raw_data_path='video_example/modelling/train/Fight/'
target_path='output'

def process_data(raw_data_path, target_path,grayscale=False,getopticalflow=False, n_videos=3):
    '''
    Reads raw data video_filepath and target path.
    This function converts videos in raw_data_path to numpy array,down-sample it,
    then save the numpy array as a .npy file in target path

    Parameters:
        raw_data_path: str
        target_path: str
    Returns:
        None
    '''
    for video in range(n_videos): #Loop through each video in raw_data_path
        video_name=os.listdir(raw_data_path)[video]
        video_path=os.path.join(raw_data_path,video_name)

        #convert each video to numpy array
        numpy_file=video_to_numpy(video_path, resize=(224,224), grayscale=grayscale, getopticalflow=getopticalflow)

        #down sample numpy array file to length 64
        numpy_file= down_sampling(numpy_file, target_frames=64)

        #save numpy array file in target path
        save_path = os.path.join(target_path, video_name+'.npy')
        numpy_file = np.uint8(numpy_file)
        np.save(save_path, numpy_file)

    return None
