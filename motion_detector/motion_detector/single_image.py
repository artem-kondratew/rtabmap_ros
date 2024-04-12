#!/usr/bin/env python3


import cv2 as cv
import numpy as np

from .submodules.yolo import Yolo
from .submodules.yolo_classes import classes_list


def main():
    model = Yolo(classes=classes_list)

    frame = cv.imread('/home/user/Documents/tum.png')

    output = model.run(frame)
    success, mask = model.merge_masks(output.masks)
    if not success:
        print('error')
        exit(1)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.normalize(mask, None, 255, 0, cv.NORM_MINMAX, cv.CV_8U)
    mask = cv.dilate(mask, kernel, iterations = 2)

    cv.imshow('mask', mask)
    cv.waitKey(0)
