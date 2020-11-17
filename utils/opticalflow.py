import numpy as np
import cv2 as cv
import copy


class OpticalFlowEngine:
    def __init__(self, detector, opticalflow_calculator, masked = True):
        '''
        detector  in ["ShiTomasi"]
        opticalflow_calculator in ["Lucas-Kanade"]
        @mask: for drawing purposes, default true
        '''
        self.detector_params = None
        self.opticalflow_params = None
        self.detector = detector
        self.opticalflow_calculator = opticalflow_calculator

        # image
        self.old_gray = np.array([])
        self.new_gray = np.array([])

        # mask
        self.masked = masked
        self.mask = None
        # corners
        self.corners = []
        self.good_corners = []
        self.good_new_corners = []
    
    def set_detection_params(self, params):
        self.detector_params = params

    def detect_corners(self, old_gray):
        '''
        detect the stringest N corners in an image
        '''
        assert old_gray.size != 0, "not valid image"
        self.old_gray = old_gray

        # create mask if required
        if self.masked: self._reset_mask()

        corners = []
        if self.detector == "ShiTomasi":
            corners = cv.goodFeaturesToTrack(old_gray, mask = None, **self.detector_params)
            self.corners = corners
        return corners

    def set_opticalflow_params(self, lk_params):
        self.opticalflow_params = lk_params

    def compute_new_corners(self, new_gray):
        '''
        detect new positions of corners computed in
        a previous step
        '''
        assert self.old_gray.size != 0, "The old image is not found"
        assert self.old_gray.shape == new_gray.shape, "images shapes don't match"
        self.new_gray = new_gray
        # detect new corners
        next_corners, st, _ = cv.calcOpticalFlowPyrLK(
            self.old_gray, # first 8-bit image
            new_gray, # second 8-bit image
            self.corners, # list of corners: [], for which flow needs to be calculated
            None, # 
            **self.opticalflow_params)
        # Select good points
        self.good_new_corners = next_corners[st==1]
        self.good_corners = self.corners[st==1]

        return self.good_new_corners

    def draw_tracks(self, frame):
        '''
        draw on the mask lines and points on the original image
        then merge them
        '''
        # deep copy
        img = np.copy(frame)
        if self.masked:
            assert img.shape[:2] == self.mask.shape[:2], "mask and img shapes don't match "
            # ensure they are the same dtype
            self.mask = self.mask.astype(img.dtype)
        for _,(new,old) in enumerate(zip(self.good_new_corners, self.good_corners)):
            a,b = new.ravel()
            c,d = old.ravel()
            if self.masked:
                self.mask = cv.line(self.mask, (a,b),(c,d), [0, 255, 255], 3)
            img = cv.circle(img,(a,b),5,[0, 0, 0],-1)
        
        # adding mask to the original image
        opt_img = img
        if self.masked:
            opt_img = cv.add(opt_img,self.mask)

        return opt_img

    def _reset_mask(self):
        '''
        reset mask to zero numpy array
        '''
        self.mask = np.zeros((self.old_gray.shape + (3,)))

    def update(self):
        '''
        update old_gray, corners. 
        To be called after the method draw_tracks,
        but not neccessary, otherwise you lose data
        '''
        self.old_gray = self.new_gray
        self.corners = self.good_new_corners.reshape(-1,1,2)




