import numpy as np
from Fightdet.video_to_numpy import video_to_numpy
from Fightdet.down_sample import down_sampling

def process_data(video_path, grayscale=True,optical_only=False, n_videos=1):
    '''
    Loads a video at video_path as an array and reduces the size of the array using downsampling and downcasting
    Parameters:
        video_path: str
        grayscale: bool
        getopticalflow: bool
        n_videos: int
    Returns:
        ndarray of shape (frame, height, width, channels)
    '''

    # convert each video to ndarray
    numpy_file=video_to_numpy(video_path, resize=(224,224), grayscale=grayscale, optical_only=optical_only)
    # downsampling to 64 frames and downcasting to integer, to reduce the memory size of the array
    numpy_file= down_sampling(numpy_file, target_frames=64)
    numpy_file = np.uint8(numpy_file)

    return numpy_file
