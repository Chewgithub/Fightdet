import numpy as np

def normalize(data):
    '''
    Applies normalization (standard scaling) to all values in an array.

    Parameters:
        data: ndarray
    Returns:
        ndarray
    '''
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean) / std

def load_data(numpy_file):
    '''
    Loads a numpy array and applies normalization to it.

    Parameters:
        numpy_file: ndarray of shape (frame, height, width, channels)
    Returns:
        ndarray of shape (frame, height, width, channels)
    '''
    data = np.float32(numpy_file)
    data[...,:1] = normalize(data[...,:1])
    data[...,1:] = normalize(data[...,1:])
    return data
