from Fightdet.pre_processing import process_data
from Fightdet.video_to_npy import load_data
# from tensorflow.keras.models import load_model

def make_prediction(model, video_path):
    # model=load_model('final_model(gray_op)')
    numpy_file=process_data(video_path, grayscale=True,optical_only=False, n_videos=1)
    pre_processed_numpy=load_data(numpy_file)

    result=model.predict(pre_processed_numpy.reshape(1,64,224,224,3))[0][0]

    return result
