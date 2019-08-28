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
    img = cv2.imread(os.path.join(img_file,img_file_list[0]))
    edges = cv2.Canny(img, 400, 400)

    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def image_seg_watershed():
    img = cv2.imread(os.path.join(img_file,img_file_list[0]))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    plt.subplot(121), plt.imshow(thresh)
    plt.show()


def image_seg_skimage():
    img = io.imread((os.path.join(img_file, img_file_list[0])))
    grayscale = color.rgb2gray(img)

    global_thresh = filters.threshold_otsu(grayscale)
    binary_global = grayscale > global_thresh
    plt.imshow(binary_global)
    plt.show()


if __name__ == '__main__':
    print((os.path.join(img_file, img_file_list[0])))
    #image_seg_opencv()
    image_seg_watershed()
    #image_seg_skimage()