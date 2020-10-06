# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import cv2 as cv
import argparse
import time



parts_to_compare = [
    ('left shoulder','right shoulder'),
    ('left shoulder','left elbow'),
    ('right shoulder','right elbow'),
    ('left elbow', 'left wrist'),
    ('right elbow','right wrist'),
    ('left hip','right hip'),
    ('left shoulder','left hip'),
    ('right shoulder','right hip'),
    ('left hip','left knee'),
    ('right hip','right knee'),
    ('left knee','left ankle'),
    ('right knee','right ankle')]
    
def draw_pose(img, keypoints, pairs):
    for i, pair in enumerate(pairs):
        color = (0,255,0)
        cv.line(img, (keypoints[pair[0]].yx[1], keypoints[pair[0]].yx[0]), (keypoints[pair[1]].yx[1], keypoints[pair[1]].yx[0]), color=color, lineType=cv.LINE_AA, thickness=1)



# os.system('wget https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/'
#           'Hindu_marriage_ceremony_offering.jpg/'
#           '640px-Hindu_marriage_ceremony_offering.jpg -O couple.jpg')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="tflite path")
    parser.add_argument("--video", help="video path")
    parser.add_argument("--height", help="height")
    parser.add_argument("--width", help="width")
    parser.add_argument("--seconds", help="second of the video to be played", default=5)
    args = parser.parse_args()
    
    # model path
    model_path = args.model
    # video path
    video_path = args.video
    video = cv.VideoCapture(video_path)
    print(video.isOpened())

    width = int(args.width)
    height = int(args.height)

    # pose engine
    engine = PoseEngine(model_path)

    # 1 second
    # n_sec = video.get(cv.CAP_PROP_POS_MSEC) if args.seconds == 'none' else int(args.seconds)
    n_sec = int(args.seconds)

    print(args.seconds)
    t_curr = t_start = time.time()
    print(t_curr)
    print(t_start)
    t_end = t_curr + n_sec
    print(t_end)
    # frames counter
    i = 0
    # inference time
    inf_times = []
    while video.isOpened():
        t_curr = time.time()
        if t_curr > t_end:
            break
        # process the frame here
        ret, frame = video.read()
        if ret == False:
            break
        frame = cv.resize(frame, (width, height))
        poses, inference_time = engine.DetectPosesInImage(np.uint8(frame))
        # INFERENCE TIME
        inf_times.append(inference_time)
        print('Inference time: %.fms' % inference_time)

# pil_image = Image.open('couple.jpg')
# pil_image.resize((641, 481), Image.NEAREST)
# poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
# print('Inference time: %.fms' % inference_time)



# template_show = np.array(pil_image.convert('RGB'))
# template_show = template_show[:, :, ::-1].copy()

        template_show = np.array(frame)
        for pose in poses:
            if pose.score < 0.4: continue
            print('\nPose Score: ', pose.score)
            # print(pose.keypoints)
            # pose.keypoints
            draw_pose(template_show, pose.keypoints, parts_to_compare)
            # cv.imshow('Frame',template_show)
            # cv.waitKey(0)

            # for label, keypoint in pose.keypoints.items():
            #     print(' %-20s x=%-4d y=%-4d score=%.1f' %
            #         (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))

        cv.imshow('Frame',template_show)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        i+=1
    
    print("Frames per second for the video: {0} fps".format(video.get(cv.CAP_PROP_FPS)))
    print("inference performance for the model: {0} fps".format(i/(t_curr - t_start)))
    cv.waitKey(0)
    video.release()
    cv.destroyAllWindows()




# parts to compare
#parts_to_compare = [(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(5,11),(6,12),(11,13),(12,14),(13,15),(14,16)]


# KEYPOINTS = (
#   0'nose',
#   1'left eye',
#   2'right eye',
#   3'left ear',
#   4'right ear',
#   5'left shoulder',
#   6'right shoulder',
#   7'left elbow',
#   8'right elbow',
#   9'left wrist',
#   10'right wrist',
#   11'left hip',
#   12'right hip',
#   13'left knee',
#   14'right knee',
#   15'left ankle',
#   16'right ankle'
# )