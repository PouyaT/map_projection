import os
import cv2
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
import statistics
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio
from skimage.transform import resize

map_size = 200

counter = 0
scale = -1
divider = 10

img_file = os.path.join(os.getcwd(), "Input_Images")
img_file_list = os.listdir(img_file)

video_file = os.path.join(os.getcwd(), "Input_Videos")
video_file_list = os.listdir(video_file)

# use this as a queue. enqueue: list.append() deque: list.pop(0)
glob_avg_x_loc = list()
glob_avg_y_loc = list()

grass_image = False


# returns a filtered image and unfiltered image. This is needed for white lines on green grass
# output are two images, First output is the filtered image, Second output is the original pre-filtered image
def grass_filter(og_image):
    kernel = np.ones((3, 3), np.uint8)

    result = og_image.copy()
    img = cv2.cvtColor(og_image, cv2.COLOR_BGR2HSV)
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

    plt.imshow(result)
    plt.show()

    # cv2.imshow('mask', mask)
    # cv2.imshow('result', result)
    # cv2.imshow('opening_test', gradient)
    # cv2.waitKey()

    return result, og_image


# takes in an image and outputs an image that has redlines overlaying the detected boundries
def pipeline(image):

    # flag for seeing if you're dealing with grassy images or not
    if grass_image:
        # for white lines on grassy photos need this part for pre-processing before being able to detect white lines
        img_list = grass_filter(image)
        pre_or_post_filtered_image = img_list[0]
        original_overlay_image = img_list[1]
    else:
        # need this if testing on actual roads with white lines borders
        pre_or_post_filtered_image = image
        original_overlay_image = image

    # gives the height and width of the image from the dimensions given
    height = image.shape[0]
    width = image.shape[1]

    # curvy road test width-50
    # region_of_interest_vertices = [
    #     (0, height),
    #     (width / 2 - 10, height / 2 + 50),
    #     (width - 50, height),
    # ]

    # using for the hd highway video to for propper crop
    # region_of_interest_vertices = [
    #     (300, height),
    #     (width / 2 + 100, height / 2 + 300),
    #     (width * .8, height),
    # ]

    # used for non-hd video
    region_of_interest_vertices = [
        (0, height),
        (width / 2, height / 2 + 70),
        (width, height),
    ]

    # convert to grayscale
    gray_image = cv2.cvtColor(pre_or_post_filtered_image, cv2.COLOR_RGB2GRAY)

    # call Canny Edge Detection
    cannyed_image = cv2.Canny(gray_image, 100, 200)

    # crop operation at the end of the cannyed pipeline so cropped edge doesn't get detected
    cropped_image = region_of_interest(cannyed_image, np.array([region_of_interest_vertices], np.int32))

    # used houghlinesP algo to detect the white lines
    # use threshold=152 for road side image. Test out different stuff for grassy images
    lines = cv2.HoughLinesP(cropped_image, rho=6, theta=np.pi / 60, threshold=75,
                            lines=np.array([]), minLineLength=40, maxLineGap=25)

    line_image = draw_lines(original_overlay_image, lines)

    # gets the centered x and y location of the current frame
    # currently not used
    # frame_x_loc, frame_y_loc = current_x_n_y_loc(lines)

    map_localization(lines, width, height)

    # # this is to display images for testing purpose
    # plt.figure()
    # plt.imshow(line_image)
    # plt.show()

    return line_image


# takes in an image and a list of points (vertices) to crop the image
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


# teaks in an image, a list of lines from HoughedLinesP and draws red lines on top of the passed in image
# outputs the images
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


# testing out cropping mechanism
def crop_images():

    # read in desired image
    image = mpimg.imread(os.path.join(img_file, img_file_list[4]))

    # images I'm using are 540x960x3
    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width / 2 - 25, height / 2 + 50),
        (width - 175, height),
    ]

    # region_of_interest_vertices = [
    #     (0, height),
    #     (width / 2 - 150, height / 2 + 200),
    #     (width - 450, height),
    # ]

    plt.figure()
    plt.imshow(image)
    plt.show()

    # convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # crop operation at the end of the cannyed pipeline so cropped edge doesn't get detected
    cropped_image = region_of_interest(gray_image, np.array([region_of_interest_vertices], np.int32))

    plt.figure()
    plt.imshow(cropped_image)
    plt.show()

    # takes in a list of coordinates that are lines


# takes in a list of lines and figures out the current x loc and y loc of the frame
# not needed for SLAM for now
# TODO Currently not in use
def current_x_n_y_loc(lines):
    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            # using the slope might not work for curved lanes. Might just need to split the image in half at
            # an x pixel point and everything to the left is left side and everything to the right is right side
            slope = (y2 - y1) / (x2 - x1)  # <-- Calculating the slope.
            if slope <= 0:  # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # finds the median x and y coordinates of each side of the line
    try:
        right_line_x_med = statistics.median(right_line_x)
        right_line_y_med = statistics.median(right_line_y)
        left_line_x_med = statistics.median(left_line_x)
        left_line_y_med = statistics.median(left_line_y)

        # gets min values of each x/y coordinates of each sides of the line
        right_line_min_x = min(right_line_x)
        right_line_min_y = min(right_line_y)
        left_line_min_x = min(left_line_x)
        left_line_min_y = min(left_line_y)

        # distance between lines
        x_dist = math.fabs(right_line_min_x - left_line_min_x)
        y_dist = math.fabs(right_line_min_y - left_line_min_y)

        # get the average value of the min values for left and right line
        # may need to add a counterweight with median value on a more consistent line set
        min_x_avg = (right_line_min_x + left_line_min_x) / 2
        min_y_avg = (right_line_min_y + left_line_min_y) / 2
    except statistics.StatisticsError:
        # returns a -1 which is normally impossible to return
        return -1, -1

    return min_x_avg, min_y_avg


# takes in the current x and y location and sees if they differ from the avg location from the previous 15
# frames to see if the a turn is required or not
# not actually related right now
# TODO test out this method idk how accurate this is if at all
# TODO Currently not in use
def desired_loc(curr_x_loc, curr_y_loc):
    # gets current angle (in degrees)
    curr_angle = np.arctan2(curr_x_loc, curr_y_loc) * (180 / np.pi)

    # catches the -1 and returns nothing and ends the function
    if curr_x_loc == -1 or curr_y_loc == -1:
        return None
    # makes sure there ar at least 15 values in the list before doing turn evaluation
    if len(glob_avg_x_loc) < 15 and len(glob_avg_y_loc) < 15:
        glob_avg_x_loc.append(curr_x_loc)
        glob_avg_y_loc.append(curr_y_loc)
        print('list to short')
    # print out which direction the robot should be going and update the lit (que)
    else:
        avg_x = statistics.mean(glob_avg_x_loc)
        avg_y = statistics.mean(glob_avg_y_loc)

        # gets the average angle of the frames
        avg_angle = np.arctan2(avg_x, avg_y) * (180 / np.pi)

        if curr_angle > avg_angle:
            print('turn right dif is: ', curr_angle - avg_angle)
            glob_avg_x_loc.pop(0)
            glob_avg_x_loc.append(curr_x_loc)
            glob_avg_y_loc.pop(0)
            glob_avg_y_loc.append(curr_y_loc)
        elif curr_angle < avg_angle:
            print('turn left dif is: ', curr_angle - avg_angle)
            glob_avg_x_loc.pop(0)
            glob_avg_x_loc.append(curr_x_loc)
            glob_avg_y_loc.pop(0)
            glob_avg_y_loc.append(curr_y_loc)


# the if statement that determines what is left or right lane will need to change based on video footage
def map_localization(lines, width, height):
    global map_size
    global scale
    global divider

    x_left_list = []
    x_right_list = []
    y_left_list = []
    y_right_list = []

    x_left_list_r = []
    x_right_list_r = []
    y_left_list_r = []
    y_right_list_r = []

    # r stands for rounded value
    # I add 50 to guarantee that it will round up to the neared hundreds
    width_r = int(round_up(width, scale))
    height_r = int(round_up(height, scale))
    height_r = height_r / divider
    width_r = width_r / divider

    # populates two lists with x and y values
    for line in lines:
        for x1, y1, x2, y2 in line:
            # if the x location of the pixel is less than 500 then it's the left line
            # this if statement will need to change based on video footage
            if x1 < 500:
                x_left_list.extend([x1, x2])
                y_left_list.extend([y1, y2])
            else:
                x_right_list.extend([x1, x2])
                y_right_list.extend([y1, y2])

    # makes sure the list isn't empty before trying to round
    # creating np arrays from original right and left line lists
    # then using np's built in functions to round
    # and turning them back into normal lists
    # work on this ,maybe go back to what i had
    if len(x_right_list) > 0:
        x_right_list_r = list(np.array(x_right_list) / divider)
        y_right_list_r = list(np.array(y_right_list) / divider)

    if len(x_left_list) > 0:
        x_left_list_r = list(np.array(x_left_list) / divider)
        y_left_list_r = list(np.array(y_left_list) / divider)

    # creates a 2d array of 6x10 (in this particular case) (row x column)
    data_map = np.zeros(shape=(int(height_r), int(width_r)), dtype=int)

    # error checking
    if len(x_right_list) > 0:
        # gets the average of the x on the left and right sides
        # I'm doing this so I can get a consistent line
        x_right_list_avg = statistics.mean(x_right_list_r)
        # loops through the x and y coordinates and places a 1 on the map representing the line from the image
        for row in y_right_list_r:
            data_map[int(row)][int(x_right_list_avg)] = 1

            for i in range(4):
                # checks i below and above to fill in any gaps that might have been missed
                if 0 < int(row) - i and data_map[int(row)][int(x_right_list_avg)] == 1:
                    data_map[int(row) - i][int(x_right_list_avg)] = 1

                if int(row) + i < height_r and data_map[int(row)][int(x_right_list_avg)] == 1:
                    data_map[int(row) + i][int(x_right_list_avg)] = 1

    # populates left side of the map
    if len(x_left_list) > 0:
        x_left_list_avg = statistics.mean(x_left_list_r)
        for row in y_left_list_r:
            data_map[int(row)][int(x_left_list_avg)] = 1

            for i in range(4):
                # checks i below and above to fill in any gaps that might have been missed
                if 0 < int(row) - i and data_map[int(row)][int(x_left_list_avg)] == 1:
                    data_map[int(row) - i][int(x_left_list_avg)] = 1

                if int(row) + i < height_r and data_map[int(row)][int(x_left_list_avg)] == 1:
                    data_map[int(row) + i][int(x_left_list_avg)] = 1
                    
    # changes the map so it's more keen on how humans read maps. The original numpy array has 0,0 as the top left corner
    plt.imshow(data_map, extent=(0, data_map.shape[1], 0, data_map.shape[0]))
    plt.show()

    data_map_resized = resize(data_map, (map_size, map_size))
    plt.imshow(data_map, extent=(0, data_map_resized.shape[1], 0, data_map_resized.shape[0]))
    plt.show()

    # # this aves all the images to the particular file directory
    # global counter
    # fig = plt.figure()
    # plt.imshow(data_map, extent=(0, data_map.shape[1], 0, data_map.shape[0]))
    # image_file_name = 'Map_Images/MAP' + str(counter)
    # plt.savefig(image_file_name)
    # plt.close(fig)
    # counter += 1


# turning the Map_Images directory into a set a video
def frames_to_videos():
    map_file = os.path.join(os.getcwd(), "Map_Images")
    map_file_list = os.listdir(map_file)

    writer = imageio.get_writer('test.mp4', fps=27)

    for im in map_file_list:
        writer.append_data(imageio.imread(os.path.join(map_file, im)))
    writer.close()

# takes in height, angle of the camera, and the field of view so the image can given a reference of a distance
def length_to_ground(height, angle, field_of_view):
    while(1):
        break


# created my own helper function to round up numbers
def round_up(n, decimals):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier


if __name__ == '__main__':
    # white_line_on_roads(resized)
    # image = mpimg.imread(os.path.join(img_file, img_file_list[5]))
    # pipeline(image)
    # crop_images()
    # images = grass_image()

    video = os.path.join(video_file, video_file_list[3])
    print(video)
    white_output = 'Highway_Video_with_cars.mp4'
    clip1 = VideoFileClip(video)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(white_output, audio=False)

    # frames_to_videos()

