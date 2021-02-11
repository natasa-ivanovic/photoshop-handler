import cv2
import numpy as np
import time

# global params #
# image saving settings
img_counter = 500
save_images = True
gesture_name = 'nav'

# size params for window - 0, 1 would be entire screen
cap_region_x_begin = 0.5
cap_region_y_end = 0.8

# threshold and blur value for image processing
threshold = 60
blur_value = 41
# blur_value = 61

# sets threshold and LR for background subtractor
bg_threshold = 70
# bg_threshold = 50
# bg_threshold = 16
bg_learning_rate = 0

# sets minimum size of contour to be considered valid
area_limit = 2000

# debug mode - show all windows
debug_mode = False


def process_frame(frame, bg_model):
    fg_mask = bg_model.apply(frame, learningRate=bg_learning_rate)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
    img = cv2.bitwise_and(frame, frame, mask=fg_mask)
    # crop photo to constraints
    img = img[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    # convert the image into binary image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

    # adaptive threshold - thresh gaussian
    # source img, max value, adaptive method, threshold type, block size, constant subtracted from the mean
    # radi kilavo, ukoliko dovoljno priblizimo ruku uspe da donekle uradi segmentaciju
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 105, 10)

    # adaptive threshold - thresh mean
    # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 20)

    # combo thresh binary and otsu - najbolje radi
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    if debug_mode:
        cv2.imshow('mask', img)
        cv2.imshow('blur', blur)
        cv2.imshow('thresh', thresh)

    return img, thresh


def process_contours(contours, img):
    center = []
    drawing = []
    length = len(contours)
    max_area = -1
    if length > 0:
        # find the biggest contour (according to area)
        for i in range(length):
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > max_area:
                max_area = area
                ci = i
        res = contours[ci]
        if cv2.contourArea(res) < area_limit:
            return center, drawing
        # print(cv2.contourArea(res))
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        # centre of hand
        moments = cv2.moments(res)
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        cv2.circle(drawing, center, 5, (255, 0, 0), 2)
    return center, drawing


def capture_on_click(event, x, y, flags, params):
    global save_images
    drawing, thresh, img = params
    if event == cv2.EVENT_LBUTTONDOWN and save_images and drawing and thresh and img:
        capture_frame(drawing, thresh, img)


def capture_on_keystroke(drawing, thresh, img):
    global save_images
    if save_images and drawing and thresh and img:
        capture_frame(drawing, thresh, img)


def capture_frame(drawing, thresh, img):
    global gesture_name, img_counter
    img_name=f"./dataset/contours/{gesture_name}_{img_counter}.jpg"
    cv2.imwrite(img_name, drawing)
    print("{} written".format(img_name))

    img_name2 = f"./dataset/thresholds/{gesture_name}_{img_counter}.jpg"
    cv2.imwrite(img_name2, thresh)
    print("{} written".format(img_name2))

    img_name3 = f"./dataset/masks/{gesture_name}_{img_counter}.jpg"
    cv2.imwrite(img_name3, img)
    print("{} written".format(img_name3))

    img_counter += 1


def main_loop():
    bg_captured = False

    camera = cv2.VideoCapture(0)
    camera.set(10, 200)

    while camera.isOpened():
        drawing, thresh, img = [], [], []

        ret, frame = camera.read()
        # smoothing filter
        frame = cv2.bilateralFilter(frame, 5, 50, 100)

        # flip the frame horizontally
        # frame = cv2.flip(frame, 1)

        # green rectangle
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                      (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.imshow('original', frame)

        if bg_captured:
            img, thresh = process_frame(frame, bg_model, bg_learning_rate)
            # check if this is actually nececary
            # thresh1 = copy.deepcopy(thresh)
            # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            center, drawing = process_contours(contours, img)

        cv2.setMouseCallback('original', capture_on_click, params=(drawing, thresh, img))

        k = cv2.waitKey(10)

        # exit on escape
        if k == 27:
            break

        # capture the background on b
        elif k == ord('b'):
            bg_model = cv2.createBackgroundSubtractorMOG2(0, bg_threshold, False)
            bg_captured = True
            print('Background captured')

        # reset the background on r
        elif k == ord('r'):
            time.sleep(1)
            bg_model = None
            bg_captured = False
            print('Reset background')

        # capture picture on space if save_image
        elif k == 32:
            capture_on_keystroke(drawing, thresh, img)


if __name__ == '__main__':
    main_loop()
