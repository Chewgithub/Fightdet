import numpy as np
import cv2
from Fightdet.ysget_optical_flow import get_optical_flow

def video_to_numpy(video_file_path, resize=(224,224), grayscale=False, optical_only=False):
    '''
    Reads a video at video_filepath and converts it to a 4-D ndarray with dimensions (frame, height, width, channel).
    If grayscale is True, returns the ndarray with 1 grayscale channel and 2 optical flow channels, else returns ndarray with 3 RGB channels and 2 optical flow channels.
    If optical_only is True, returns the ndarray with 2 optical flow channels only.
    Parameters:
        video_file_path: str
        resize: tuple of (int, int)
        grayscale: bool
        optical_only: bool
    Returns:
        ndarray of (frame, height, width, channel)
    '''

    cap = cv2.VideoCapture(video_file_path)

    # initialize the video as an empty list
    video_array = []
    gray_video_array=[]

    while cap.isOpened():
        # read next frame
        ret, frame = cap.read()
        # if no frame is read, close the video and exit
        if not ret:
            cap.release()
            break
        # resize the video if a resize shape is given
        if resize is not None:
            frame = cv2.resize(frame, dsize=resize, interpolation=cv2.INTER_AREA)
        # change frame to RGB and add to video
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_array.append(rgb_frame)
        # make a copy of the frame in grayscale and add to gray_scale video
        if grayscale:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, -1)
            gray_video_array.append(gray_frame)

    # convert list of frames to ndarray
    video_array = np.array(video_array)
    if grayscale:
        gray_video_array = np.array(gray_video_array)
    # get_optical_flow always uses the RGB video array
    optical_flow_array = get_optical_flow(video_array)

    if optical_only:
        # return optical_flow array (150,224,224,2)
        return optical_flow_array
    elif grayscale:
        result = np.zeros((len(gray_video_array),224,224,3))
        result[...,:1] = gray_video_array
        result[...,1:] = optical_flow_array
        # return gray+optical_flow array (150,224,224,3)
        return result
    else:
        result = np.zeros((len(video_array),224,224,5))
        result[...,:3] = video_array
        result[...,3:] = optical_flow_array
        # return rgb+optical_flow array (150,224,224,5)
        return result

if __name__=="__main__":
    import os.path
    filepath = os.path.join(os.path.dirname(__file__), '..', 'raw_data',
                            '_q5Nwh4Z6ao_0.avi')
    vid = video_to_numpy(filepath, resize=(224, 224), optical_only=True)
    print("Optical flow only", vid.shape)
    vid = video_to_numpy(filepath, resize=(224, 224), optical_only=False, grayscale=True)
    print("Grayscale with optical flow", vid.shape)
    vid = video_to_numpy(filepath, resize=(224, 224), optical_only=False, grayscale=False)
    print("RGB with optical flow", vid.shape)
