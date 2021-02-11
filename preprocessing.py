import copy
import cv2
import numpy as np
import time

# save image
img_counter = 251

# parameters
cap_region_x_begin = 0.5
cap_region_y_end = 0.8
threshold = 60
blur_value = 41
#blur_value = 61
#bg_threshold = 50
#bg_threshold = 16
bg_threshold = 70
learning_rate = 0
area_limit = 10000

bg_captured = False
trigger_switch = False

save_images = True
gesture = 'delete'

def remove_background(frame):
    fgmask = bgModel.apply(frame, learningRate=learning_rate)
    kernel = np.ones((3, 3), np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res = cv2.bitwise_and(frame, frame, mask=fgmask)
    return res

camera = cv2.VideoCapture(0)
camera.set(10, 200)

while camera.isOpened():
    ret, frame = camera.read()
    # smoothing filter
    frame = cv2.bilateralFilter(frame, 5, 50, 100)
    # flip the frame horizontally
    frame = cv2.flip(frame, 1)
    # green rectangle
    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
    cv2.imshow('original', frame)

    if bg_captured == True:
        img = remove_background(frame)
        # ROI
        img = img[0:int(cap_region_y_end * frame.shape[0]),
              int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]
        cv2.imshow('mask', img)

        # convert the image into binary image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
        cv2.imshow('blur', blur)

        # adaptive threshold - thresh gaussian
        # source img, max value, adaptive method, threshold type, block size, constant subtracted from the mean
        # radi kilavo, ukoliko dovoljno priblizimo ruku uspe da donekle uradi segmentaciju
        # thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 105, 10)

        # adaptive threshold - thresh mean
        #thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 20)

        # combo thresh binary and otsu - najbolje radi
        ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imshow('thresh', thresh)

        # contours
        thresh1 = copy.deepcopy(thresh)
        contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        length = len(contours)
        maxArea = -1
        if length > 0:
            # find the biggest contour (according to area)
            for i in range(length):
                temp = contours[i]
                area = cv2.contourArea(temp)
                if area > maxArea:
                    maxArea = area
                    ci = i
            res = contours[ci]
            if (cv2.contourArea(res) < area_limit):
                continue
            print(cv2.contourArea(res))
            hull = cv2.convexHull(res)
            drawing = np.zeros(img.shape, np.uint8)
            cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
            cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
            # centre of hand
            moments = cv2.moments(res)
            center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
            cv2.circle(drawing, center, 5, (255, 0, 0), 2)

        cv2.imshow('contours', drawing)

    k = cv2.waitKey(10)
    if k == 27:
        break

    # capture the background
    elif k == ord('b'):
        bgModel = cv2.createBackgroundSubtractorMOG2(0, bg_threshold)
        #bgModel = cv2.createBackgroundSubtractorKNN(0, 400)
        bg_captured = True
        print('Background captured')
    # reset the background
    elif k == ord('r'):
        time.sleep(1)
        bgModel = None
        trigger_switch = False
        bg_captured = False
        print('Reset background')
    elif k == 32:
        # spacebar pressed
        cv2.imshow('original', frame)
        #save image with pressing the spacebar
        if save_images:
            img_name=f"./dataset/contours/{gesture}_{img_counter}.jpg"
            cv2.imwrite(img_name, drawing)
            print("{} written".format(img_name))

            img_name2 = f"./dataset/thresholds/{gesture}_{img_counter}.jpg"
            cv2.imwrite(img_name2, thresh)
            print("{} written".format(img_name2))

            img_name3 = f"./dataset/masks/{gesture}_{img_counter}.jpg"
            cv2.imwrite(img_name3, img)
            print("{} written".format(img_name3))

            img_counter += 1
