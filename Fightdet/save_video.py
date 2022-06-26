import os
import numpy as np
from Fightdet.video_to_numpy import video_to_numpy
from Fightdet.down_sample import down_sampling

def save_video(raw_video_path,target_path, grayscale=True,optical_only=False):
    '''
    Loads a video at video_path as an array and reduces the size of the array using downsampling,
    npy file will be saved at target path
    Parameters:
        video_path: str
        grayscale: bool
        getopticalflow: bool
        n_videos: int
    Returns:
        None
    '''

    #for every folder in raw_video_path(train/test)
    for folder in os.listdir(raw_video_path):
        folder_path=os.path.join(raw_video_path,folder)

        #for every class in folder_path(Fight/NonFight)
        for classes_ in os.listdir(folder_path):
            class_folder=os.path.join(folder_path,classes_)

            #for every video file in class_folder(each video file)
            for video_file in os.listdir(class_folder):

                #extract the raw file name without extension
                 video_name_without_dot=video_file.split('.')[0]
                 each_video_path=os.path.join(class_folder,video_file)

                 # convert each video to ndarray
                 numpy_file=video_to_numpy(each_video_path, resize=(224,224),
                                           grayscale=grayscale,
                                           optical_only=optical_only)

                 # downsampling to 64 frames and downcasting to integer,
                 # to reduce the memory size of the array
                 numpy_file= down_sampling(numpy_file, target_frames=64)

                 #convert numpy to dtype=uint8
                 numpy_file = np.uint8(numpy_file)
                 save_path=os.path.join(target_path,folder,classes_,video_name_without_dot)
                 np.save(save_path,numpy_file)
                 print(f'{video_name_without_dot}.npy saved!')

    return None


if __name__=="__main__":
    #video path, path where all 5sec video are stored
    raw_video_path='raw_data/video_raw_data/'

    #target path, path where all .npy file are stored
    target_path='raw_data/npy_raw_data/'

    save_video(raw_video_path,target_path, grayscale=True,optical_only=False)
