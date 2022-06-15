# A Further Exploration on Possible Variations in Violence Detection Using RWF-2000 Dataset
## Introduction

This study is inspired by the works of Cheng, Ming ([Github](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection)) and Cai, Kunjing and Li, Ming's work in [RWF-2000: An Open Large Scale Video Database](https://arxiv.org/abs/1911.05913v3) in 2019. Further exploration based on their study is conducted by testing on several different models, with different channels and optimizers. 

We are trying out possible alternatives that might be more feasible during actual production stage, which also means that the criteria for model selection will be a combination of model size, runtime, and the model accuracy.

## File Description
<ul>
<li>Original data is in video format (AVI format from RWF-2000 dataset). The video is transformed to .npy files using python script. Each .npy file is a tensor with shape = [nb_frames, img_height, img_width, 5]. The last channel contains 3 layers for RGB components and 2 layers for optical flows (vertical and horizontal components, respectively ).</li>
<li>Fightdet contains all the converted python file(.py) for the formation of pipeline.</li>
<li>notebook contains the trial run of whole model using ipynb. The whole pipeline is tested and confirmed using google colab before launching.</li>
<li>final_model(gray_op) contains the pre-trained model built using Tensorflow Keras.</li></ul>

##  Dataset
As mentioned, this exploration is done based on the RWF-2000 dataset. RWF-2000 dataset comprised of raw surveillance videos from YouTube, sliced into clips within 5s at 30 fps, and labeled each clip as Violent or Non-Violent. The duplicated contents in both training, validation set and test set are dropped to get 2000 clips and 300,000 frames as a new data set for real-world violent behavior detection under surveillance camera.

## Problems
As the downloaded videos clips are from different surveillance cameras in public places, imaging quality of the video clips varies due to dark environment, fast movement, blurry images, etc. These variations affects the quality of the model. Here are some examples:

- Only part of the person appears in the picture


- Crowds and chaos


- Small object at far distance


- Low resolution


- Transient action
