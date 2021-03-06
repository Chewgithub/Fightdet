import numpy as np
import cv2
from Fightdet.ysget_optical_flow import get_optical_flow

def video_to_numpy(video_file_path, resize=(224,224), grayscale=True, optical_only=False):
    '''
    Reads a video at video_filepath and converts it to a 4-D ndarray with dimensions (frame, height, width, channel).
    If grayscale is True, returns the ndarray with 1 grayscale channel and 2 optical flow channels, else returns ndarray with 3 RGB channels and 2 optical flow channels.
    If optical_only is True, returns the ndarray with 2 optical flow channels only.

    Parameters:
        video_file_path: str
        resize: tuple of (int, int), default (224,224)
        grayscale: bool, default True
        optical_only: bool, default
    Returns:
        ndarray of shape (frame, height, width, channels)
    '''

    cap = cv2.VideoCapture(video_file_path)

    # list to contain video frames
    gray_video_array=[]

    while cap.isOpened():
        ret, frame = cap.read()
        # if the capture device does not read a frame, close the video
        if not ret:
            cap.release()
            break

        # resize the video if a resize shape is given
        if resize is not None:
            frame = cv2.resize(frame, dsize=resize, interpolation=cv2.INTER_AREA)

        # make a copy of the frame in grayscale and add to gray_video_array
        if grayscale:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = np.expand_dims(gray_frame, -1)
            gray_video_array.append(gray_frame)

    # construct gray+optical_flow array with shape (150,224,224,3)
    if not optical_only:
        gray_video_array = np.array(gray_video_array)
        optical_flow_array = get_optical_flow(gray_video_array)
        result = np.zeros((len(gray_video_array),224,224,3))
        result[...,:1] = gray_video_array
        result[...,1:] = optical_flow_array

    # check the shape of the resulting array
    print(result.shape)
    return result


# if __name__=="__main__":
#     import os.path
#     filepath = os.path.join(os.path.dirname(__file__), '..', 'raw_data',
#                             '_q5Nwh4Z6ao_0.avi')
#     vid = video_to_numpy(filepath, resize=(224, 224), optical_only=True)
#     print("Optical flow only", vid.shape)
#     vid = video_to_numpy(filepath, resize=(224, 224), optical_only=False, grayscale=True)
#     print("Grayscale with optical flow", vid.shape)
#     vid = video_to_numpy(filepath, resize=(224, 224), optical_only=False, grayscale=False)
#     print("RGB with optical flow", vid.shape)
