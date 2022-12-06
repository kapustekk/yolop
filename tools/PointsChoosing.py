import cv2
import numpy as np

CHOSEN_POINTS_COUNTER = 0
CHOSEN_POINTS = np.zeros((6,2), np.int)

def click_event(event,x,y,flags,param):
    global CHOSEN_POINTS_COUNTER
    global CHOSEN_POINTS

    if event == cv2.EVENT_LBUTTONDOWN: # captures left button double-click

        CHOSEN_POINTS[CHOSEN_POINTS_COUNTER] = x,y
        CHOSEN_POINTS_COUNTER = CHOSEN_POINTS_COUNTER + 1
        print(CHOSEN_POINTS)

def camera_calibration(path):
    global CHOSEN_POINTS_COUNTER
    global CHOSEN_POINTS
    cImage = cv2.imread(path)
    print(CHOSEN_POINTS)

    cv2.imshow("Camera calibrator", cImage)
    cv2.setMouseCallback('Camera calibrator', click_event)
    while CHOSEN_POINTS_COUNTER < 6:
        cv2.waitKey(1)
    cv2.destroyAllWindows()
    return CHOSEN_POINTS



