from Fightdet.pre_processing_prediction import process_data
from Fightdet.load_data_prediction import load_data

def make_prediction(model, video_path):
    '''
    Load and pre-process a video, then use fitted model to generate prediction on the processed video.
    The prediction is given as the probability (between 0 and 1) of the video being labelled "1" (violent activity detected) in a binary classification setup.

    Parameters:
        model: fitted tf.keras.Model
        video_path: str
    Returns:
        float in the range [0, 1]
    '''
    # load video at video_path as a numpy ndarray, convert to grayscale and add optical flows
    numpy_file=process_data(video_path, grayscale=True,optical_only=False, n_videos=1)
    # apply normalization to the ndarray
    pre_processed_numpy=load_data(numpy_file)

    # generate prediction as a probability
    result=model.predict(pre_processed_numpy.reshape(1,64,224,224,3))[0][0]

    return result
