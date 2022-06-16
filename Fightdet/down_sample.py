import numpy as np

def down_sampling(video_array, target_frames=64):
    '''
Reads a video array that is more than 64 frames and down sample it to 64 frames
without changing the dimensions (frame, height, width, channel)
Parameters:
    video_array: numpy array of len(video_array) more than 64

Returns:
    ndarray of (64, height, width, channel)
'''
    # get total frames of input video and calculate sampling interval
    len_frames = int(len(video_array))
    interval = int(np.ceil(len_frames/target_frames))
    # init empty list for sampled video and
    sampled_video = []
    for i in range(0,len_frames,interval):
        sampled_video.append(video_array[i])
    # calculate numer of padded frames and fix it
    num_pad = target_frames - len(sampled_video)
    padding = []
    if num_pad>0:
        for i in range(-num_pad,0):
            try:
                padding.append(video_array[i])
            except:
                padding.append(video_array[0])
        sampled_video += padding
    # get sampled video
    return np.array(sampled_video, dtype=np.float32)
