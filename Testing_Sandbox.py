import os
import cv2
import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import math
from PIL import Image
from skimage.feature import canny
from skimage import io
from skimage import color
from skimage import feature
from skimage import filters

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

def white_line_on_road_driver():
    # images I'm using are 540x960x3
    height = 540
    width = 960

    # read in desired image
    image = mpimg.imread(os.path.join(img_file, img_file_list[2]))

    # printing out some stats and plotting the image
    # print('This image is:', type(image), 'with dimensions:', image.shape)
    # plt.imshow(image)
    # plt.show()

    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2),
        (width, height),
    ]

    plt.figure()
    plt.imshow(image)
    plt.show()

    # convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # call Canny Edge Detection
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    # crop operation at the end of the cannyed pipeline so cropped edge doesn't get detected
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=152,
                            lines=np.array([]), minLineLength=40, maxLineGap=25)

    line_image = draw_lines(image, lines)

    # left_line_x = []
    # left_line_y = []
    # right_line_x = []
    # right_line_y = []
    #
    # for line in lines:
    #     for x1, y1, x2, y2 in line:
    #         slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
    #         if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
    #             continue
    #         if slope <= 0:  # <-- If the slope is negative, left group.
    #             left_line_x.extend([x1, x2])
    #             left_line_y.extend([y1, y2])
    #         else:  # <-- Otherwise, right group.
    #             right_line_x.extend([x1, x2])
    #             right_line_y.extend([y1, y2])
    #
    # min_y = image.shape[0] * (3 / 5)  # <-- Just below the horizon
    # max_y = image.shape[0]  # <-- The bottom of the image
    # poly_left = np.poly1d(np.polyfit(
    #     left_line_y,
    #     left_line_x,
    #     deg=1
    # ))
    #
    # left_x_start = int(poly_left(max_y))
    # left_x_end = int(poly_left(min_y))
    # poly_right = np.poly1d(np.polyfit(
    #     right_line_y,
    #     right_line_x,
    #     deg=1
    # ))
    #
    # right_x_start = int(poly_right(max_y))
    # right_x_end = int(poly_right(min_y))
    # line_image = draw_lines(
    #     image,
    #     [[
    #         [left_x_start, max_y, left_x_end, min_y],
    #         [right_x_start, max_y, right_x_end, min_y],
    #     ]],
    # )

    plt.figure()
    plt.imshow(line_image)

    plt.show()


def region_of_interest(img, vertices):
    # define a blank matrix that matches the iamge height/width.
    mask = np.zeros_like(img)

    # Create a match color for gray scalled images
    match_mask_color = 255

    # Fill inside the polygon
    cv2.fillPoly(mask, vertices, match_mask_color)

    # Returning the image only where mask pixels match
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image


def draw_lines(img, lines, color=[255,0,0], thickness=3):
    # If there are no lines to draw, exit
    if lines is None:
        return

    # Make a copy of the original image
    img = np.copy(img)

    # create a blank image that matches the original in size
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8,)

    # loop over all lines and draw them on the blank image
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)

    # Merge the image with the lines on the orginal
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

    # return the modified image
    return img


if __name__ == '__main__':
    #image_seg_opencv()
    white_line_on_road_driver()
