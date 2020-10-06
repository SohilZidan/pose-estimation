# Human Pose Estimation
Estimatate body poses while a video is being played using PoseNet models provided by [goagle coral](https://github.com/google-coral/project-posenet) repo.

## Requirements
* [EdgeTpu](https://coral.ai/docs/edgetpu/api-intro/)
* [Tensorflow](https://www.tensorflow.org/install) >= 2

  run:  `pip install -r requirements.txt`

## Running
`video_pose.py` can only be run with the following arguments:
* --model: the path of the tflite model
* --video: the path of the video
* --height, --width: resolution of the model

and optionally:
* --seconds: seconds to be played of the specified video
