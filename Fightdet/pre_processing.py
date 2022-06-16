import numpy as np
from Fightdet.video_to_numpy import video_to_numpy
from Fightdet.down_sample import down_sampling

def process_data(video_path, grayscale=True,optical_only=False, n_videos=1):
    '''
    Reads raw data video_filepath and target path.
    This function converts videos in raw_data_path to numpy array,down-sample it,
    then save the numpy array as a .npy file in target path

    Parameters:
        raw_data_path: str
        target_path: str
        grayscale: bool
        getopticalflow: bool
        n_videos: int
    Returns:
        None
    '''

    #convert each video to numpy array
    numpy_file=video_to_numpy(video_path, resize=(224,224), grayscale=grayscale, optical_only=optical_only)
    numpy_file= down_sampling(numpy_file, target_frames=64)
    numpy_file = np.uint8(numpy_file)

    return numpy_file
