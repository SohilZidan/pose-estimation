import numpy as np
import cv2 as cv
import argparse


# costumized
from utils.opticalflow import OpticalFlowEngine



def parse_input():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model", help="tflite path")
    parser.add_argument("--video", help="video path")
    # parser.add_argument("--height", help="height")
    # parser.add_argument("--width", help="width")
    # parser.add_argument("-o", "--output", help="output", action="store_true")
    # parser.add_argument("--seconds", help="second of the video to be played", default=5)

    return parser

if __name__ == '__main__':
    
    # parse input
    parser = parse_input()
    args = parser.parse_args()
    
    # video path
    video_path = args.video
    video = cv.VideoCapture(video_path)

    # read the first frame
    ret, frame1 = video.read()

    # Take first frame and find corners in it
    old_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    # determine strong corners on an image
    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100, # the strongest N corners specified are returned
                        qualityLevel = 0.3, # control the range of extracted corners
                        minDistance = 7, # min Euclidean dist between the returned corners
                        blockSize = 7 ) # size of block for computing a derivative covariation matrix over each pixel

    of_engine = OpticalFlowEngine(detector='ShiTomasi', opticalflow_calculator="Lucas-Kanade")
    of_engine.set_detection_params(feature_params)
    # ShiTomasi corner detector
    corners = of_engine.detect_corners(old_gray)# cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    # Create a mask image for drawing purposes
    # mask = np.zeros_like(frame1)

    # Parameters for lucas kanade optical flow calculator
    # iterative Lucas-Kanade method with pyramids
    lk_params = dict( winSize  = (15,15), # size of search window
                    maxLevel = 2, # 2-based maximal pyramid level numbers
                    # type, count, eps
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)) # termination criteria
    of_engine.set_opticalflow_params(lk_params)



    while video.isOpened():
        # process the frame here
        ret, frame = video.read()
        
        if ret == False:
            break

        # detect new corners
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        next_corners =  of_engine.compute_new_corners(frame_gray) # calc_optical_

        # draw the tracks
        opt_img = of_engine.draw_tracks(frame)

        # update
        of_engine.update()

        cv.imshow('opticalflow', frame)

        if cv.waitKey(25) & 0xFF == ord('q'):
            break
        
    
   
    video.release()
    cv.destroyAllWindows()