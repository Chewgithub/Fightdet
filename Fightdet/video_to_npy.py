import numpy as np

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data-mean) / std

def random_flip(video, prob=0.5):
    s = np.random.rand()
    if s < prob:
        video = np.flip(m=video, axis=2)
    return video

def load_data(numpy_file):
    data = np.float32(numpy_file)
    data = random_flip(data, prob=0.5)
    data[...,:1] = normalize(data[...,:1])
    data[...,1:] = normalize(data[...,1:])
    return data
