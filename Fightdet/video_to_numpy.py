import numpy as np
import cv2 as cv
from Fightdet.ysget_optical_flow import get_optical_flow as gof

def video_to_numpy(video_file_path, resize=(224,224), grayscale=False, optical_only=False):
    '''
    Reads a video at video_filepath and converts it to a 4-D ndarray with dimensions (frame, height, width, channel).
    If get_optical_flow is True, returns the ndarray with the last dimension containing the optical flows.
    Parameters:
        video_file_path: str
        resize: tuple of (int, int)
        grayscale: bool
        get_optical_flow: bool
    Returns:
        ndarray of (frame, height, width, channel)
    '''

    cap = cv.VideoCapture(video_file_path)

    # initialize the video as an empty list
    video_array = []
    gray_video_array=[]
    if optical_only:
        while cap.isOpened():
            # if no frame is read, close the video and exit
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            # resize the video if a resize shape is given
            if resize is not None:
                frame = cv.resize(frame, dsize=resize, interpolation=cv.INTER_AREA)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                # join the next frame to the video
                video_array.append(frame)
        video_array = np.array(video_array)
        optical=gof(video_array)
        return optical

    elif grayscale:
        while cap.isOpened():
            # if no frame is read, close the video and exit
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break
            # resize the video if a resize shape is given
            if resize is not None:
                frame = cv.resize(frame, dsize=resize, interpolation=cv.INTER_AREA)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                video_array.append(frame)

                grayframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                # join the next frame to the video
                gray_video_array.append(grayframe)

        gray_video_array = np.array(gray_video_array)

        optical=gof(video_array)
        result = np.zeros((len(gray_video_array),224,224,3))
        result[...,:1] = gray_video_array
        result[...,1:] = optical
        return result


    else:
        while cap.isOpened():
            # if no frame is read, close the video and exit
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break

            # resize the video if a resize shape is given
            if resize is not None:
                frame = cv.resize(frame, dsize=resize, interpolation=cv.INTER_AREA)
                frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                video_array.append(frame)

        video_array = np.array(video_array)
        optical=gof(video_array)
        result = np.zeros((len(video_array),224,224,3))
        result[...,:3] = video_array
        result[...,3:] = optical
        return result


    # # read and append additional frames to the list; resize and color if needed
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     # if no frame is read, close the video and exit
    #     if not ret:
    #         cap.release()
    #         break

    #     # resize the video if a resize shape is given
    #     if resize is not None:
    #         frame = cv.resize(frame, dsize=resize, interpolation=cv.INTER_AREA)
    #     # convert OpenCV BGR format to Grayscale or RGB

    #     if optical_only:
    #         frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #         video_array.append(frame)

    #     if grayscale:
    #         frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    #         frame = np.expand_dims(frame, -1)
    #     else:
    #         frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    #     # join the next frame to the video
    #     video_array.append(frame)

    # # convert list of frames to an array
    # video_array = np.array(video_array)

    # if get_optical_flow and grayscale:
    #     # add 2 optical flow channels
    #     optical=gof(video_array)
    #     result = np.zeros((len(video_array),224,224,3))
    #     result[...,:1] = video_array
    #     result[...,1:] = optical
    #     return result

    # elif get_optical_flow:
    #     # add 2 optical flow channels
    #     optical=gof(video_array)
    #     result = np.zeros((len(video_array),224,224,5))
    #     result[...,:3] = video_array
    #     result[...,3:] = optical
    #     return result

    # else:
    #     return video_array


if __name__=="__main__":
    import os.path
    filepath = os.path.join(os.path.dirname(__file__), '..', 'raw_data',
                            '_q5Nwh4Z6ao_0.avi')
    vid = video_to_numpy(filepath, resize=(224, 224))
    print(vid.shape)
