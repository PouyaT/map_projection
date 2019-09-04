import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.feature import canny
from skimage import io
from skimage import color
from skimage import feature
from skimage import filters
from PIL import Image

img_file = os.path.join(os.getcwd(), "Images")
img_file_list = os.listdir(img_file)


def image_seg_opencv():
    kernel = np.ones((3, 3), np.uint8)

    img = cv2.imread(os.path.join(img_file,img_file_list[0]))
    result = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # create a lower bound for a pixel value
    lower = np.array([0, 0, 200])
    # create an upper bound for a pixel values
    upper = np.array([179, 77, 255])
    # detects all white pixels wihin the range specified earlier
    mask = cv2.inRange(img, lower, upper)
    result = cv2.bitwise_and(result, result, mask=mask)

    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1:
            cv2.drawContours(result, [c], -1, (0, 0, 0), -1)

    gradient = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    cv2.imshow('mask', mask)
    cv2.imshow('result', result)
    cv2.imshow('opening_test', gradient)
    cv2.waitKey()


if __name__ == '__main__':
    image_seg_opencv()
