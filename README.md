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

<table align='center'>
  <tr>
    <th colspan="2" style="text-align:center">Problems</th>
  </tr>
  <tr>
    <td align="center">Only part of the person appears in the picture</td>
    <td align="center">Crowds and chaos</td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/Chewgithub/Fightdet/blob/master/images/blocked.gif" width="400px" height="250px"></td>
    <td align="center"><img src="https://github.com/Chewgithub/Fightdet/blob/master/images/crowded.gif" width="400px" height="250px"></td>
  </tr>
  <tr>
    <td align="center">Small object at far distance</td>
    <td align="center">Low resolution</td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/yongsin91/Fightdet/blob/master/images/far_distance.gif" width="400px" height="250px"></td>
    <td align="center"><img src="https://github.com/Chewgithub/Fightdet/blob/master/images/low_resolution.gif" width="400px" height="250px"></td>
  </tr>
  <tr>
    <td align="center">Transient action</td>
    <td></td>
  </tr>
  <tr>
    <td align="center"><img src="https://github.com/Chewgithub/Fightdet/blob/master/images/transient.gif" width="400px" height="250px"></td>
    <td></td>
  </tr>
</table>
<p align='center'>Table 1 : List of limitations of the training videos that reduces model accuracy</p>

## Result
We have 3 types of inputs for this model, namely RGB, grayscale and optical flows. Table below shows the result of the initial test.

<table align='center'>
  <tr>
    <td align="center"><b>Model Type</b></td>
    <td align="center"><b>Accuracy</b></td>
  </tr>
  <tr>
    <td align="center">RGB + Optical Flows</td>
    <td align="center">73%</td>
  </tr>
  <tr>
    <td align="center">Grayscale + Optical Flows</td>
    <td align="center">68%</td>
  </tr>
  <tr>
    <td align="center">Optical Flows only</td>
    <td align="center">60%</td>
  </tr>
</table>
<p align='center'>Table 2 : Initial Test conducted with 300 Training, 60 Validation and 60 Test Videos</p>

We choose the **Grayscale + Optical Flow** model as the best result because it processes 25% faster than RGB + optical flows although it has 5% lesser accuracy. In our opinion the 25% decrease in evaluate speed will be too much of an impact to be compensated while the 5% accuracy gap can become smaller after tuning the hyperparamaters and optimizers. After tuning, our final model managed to reached **84%** accuracy with the Final Training Size of 1600 and Evaluation Size of 400. Average processing time taken to predict a 5 second video is around 5 seconds. It is a 3.5% decrease from our benchmark accuracy of **87.5%**, which we considered still acceptable since the gain in evaluation speed is more significant. 

We created an simple frontend interface with streamlit, created an docker image and pushed it to Google Cloud for demonstration purposes. After testing out the model as per shown sample clips below, it seems to have perform well when tested with several new & unseen clips. The model ideally will be predicting violence video to be as close to 100% as possible while predicting non-violence to be as close to 0% as possible.

## Sample Interface Example
<table><tr><td>
<img src="https://github.com/Chewgithub/Fightdet/blob/master/images/Predict_1.gif" width="700px" height="400px"></td></tr></table>
<p align='center'>Figure 1 : The model is able to identify the violence in the video clip with 85.98% certainty.</p>


<table><tr><td>
<img src="https://github.com/Chewgithub/Fightdet/blob/master/images/Predict_0.gif" width="700px" height="400px"></td></tr></table>
<p align='center'>Figure 2 : The model is able to identify the video clip has only 26.03% probability to be violence</p>

## Conclusion
Currently this model is designed intentionally to help out for flagging violence activity in CCTV. It is able to reach 84% accuracy with 5 second processing time. However in attempting to test out the model against normal videos, we identified several limitations of the current model. As the training set is based on CCTV only, the model will not be able to identify correctly facing situations like dancing or street flash mob, as people are moving quickly, or when the camera view is moving at high speed, because the given training CCTVs videos are all static.

### The sample model website can be accessed through this link ([Fightdet Sample Interface](https://fightdet-app-gf34ldcmyq-de.a.run.app/))
