import cv2
import numpy as np
import time
import photoshop.api as ps
import pyautogui as pag

# global params #
# image saving settings
img_counter = 251
final_img_count = 500
save_images = True
gesture_name = 'nav'

# size params for window - 0, 1 would be entire screen
cap_region_x_begin = 0.5
cap_region_y_end = 0.8
#cap_region_x_begin = 0
#cap_region_y_end = 1

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

# screen size
screen_size = pag.size()

# debug mode - show all windows
debug_mode = True


# functions for picture capture
def capture_on_click(event, x, y, flags, params):
    drawing, thresh, img, bounding_box = params
    if event == cv2.EVENT_LBUTTONDOWN and save_images:
        capture_frame(drawing, thresh, img, bounding_box)


def capture_on_keystroke(drawing, thresh, img, bounding_box):
    if save_images:
        capture_frame(drawing, thresh, img, bounding_box)


def capture_frame(drawing, thresh, img, bounding_box):
    global img_counter
    if img_counter > final_img_count:
        print ("GOTOVO")
        return
    if not bounding_box:
        return
    x, y, w, h = square_bounding_box(bounding_box)
    crop_drawing = drawing[y: y+h, x:x+w]
    crop_thresh = thresh[y: y+h, x:x+w]
    crop_img = img[y: y+h, x:x+w]
    img_name = f"./dataset/contours/{gesture_name}_{img_counter}.jpg"
    cv2.imwrite(img_name, crop_drawing)
    print("{} written".format(img_name))

    img_name2 = f"./dataset/thresholds/{gesture_name}_{img_counter}.jpg"
    cv2.imwrite(img_name2, crop_thresh)
    print("{} written".format(img_name2))

    img_name3 = f"./dataset/masks/{gesture_name}_{img_counter}.jpg"
    cv2.imwrite(img_name3, crop_img)
    print("{} written".format(img_name3))

    img_counter += 1


def square_bounding_box(rectangle):
    x, y, w, h = rectangle
    if w > h:
        y -= int((w - h) / 2)
        h += int(w - h)
    else:
        x -= int((h - w) / 2)
        w += int(h - w)
    return x, y, w, h


# functions for picture processing
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
    center = ()
    drawing = []
    bounding_box = ()
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
            return center, drawing, bounding_box
        # print(cv2.contourArea(res))
        bounding_box = cv2.boundingRect(res)
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)
        # centre of hand
        moments = cv2.moments(res)
        center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
        cv2.circle(drawing, center, 5, (255, 0, 0), 2)
        if debug_mode:
            cv2.imshow('contours', drawing)
    return center, drawing, bounding_box


# functions for photoshop controll
def center_cursor():
    pag.moveTo(screen_size.width/2, screen_size.height/2)


def photoshop_setup():
    app = ps.Application()
    doc = app.documents.add(screen_size.width, screen_size.height)

    # fit doc to screen
    app.runMenuItem(app.charIDToTypeID("FtOn"))

    new_doc = doc.artLayers.add()
    new_text_layer = new_doc
    new_text_layer.kind = ps.LayerKind.NormalLayer

    # timeout so everything has time to load
    # might have to increase depending on how fast PS boots on your system
    time.sleep(3)

    # fullscreen and switch to brush
    pag.press('f', presses=2, interval=1)
    pag.press('b')

    return app, doc


# main app loop
def main_loop():
    bg_captured = False

    #photoshop, document = photoshop_setup()
    start_cursor = True
    previous_cursor = (screen_size.width/2, screen_size.height/2)
    current_cursor = previous_cursor

    # for capturing images
    bounding_box = []

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
        if debug_mode:
            cv2.imshow('original', frame)

        if bg_captured:
            img, thresh = process_frame(frame, bg_model)
            # check if this is actually nececary
            # thresh1 = copy.deepcopy(thresh)
            # contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            center, drawing, bounding_box = process_contours(contours, img)
            if center:
                if start_cursor:
                    current_cursor = center
                    previous_cursor = center
                    pag.moveTo(current_cursor[0], current_cursor[1])
                    start_cursor = False
                else:
                    cursor_difference = (previous_cursor[0] - center[0], -previous_cursor[1] + center[1])
                    previous_cursor = current_cursor
                    current_cursor = center
                    pag.moveRel(cursor_difference[0], cursor_difference[1])
                    # pag.mouseDown()

        cv2.setMouseCallback('original', capture_on_click, (drawing, thresh, img, bounding_box))

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
            capture_on_keystroke(drawing, thresh, img, bounding_box)


if __name__ == '__main__':
    main_loop()
