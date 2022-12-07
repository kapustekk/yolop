import cv2
import numpy as np
import os.path
from numpy import genfromtxt

CHOSEN_POINTS_COUNTER = 0
CHOSEN_POINTS = np.zeros((6,2), np.int)


def find_optic_middle(CHOSEN_POINTS):
    xb = int((CHOSEN_POINTS[0][0]+CHOSEN_POINTS[1][0])/2)
    yb = int((CHOSEN_POINTS[0][1]+CHOSEN_POINTS[1][1])/2)
    xu = int((CHOSEN_POINTS[2][0]+CHOSEN_POINTS[3][0])/2)
    yu = int((CHOSEN_POINTS[2][1]+CHOSEN_POINTS[3][1])/2)

    bottom_middle = [xb,yb]
    upper_middle = [xu,yu]
    return bottom_middle, upper_middle
def click_event(event,x,y,flags,param):
    global CHOSEN_POINTS_COUNTER
    global CHOSEN_POINTS

    if event == cv2.EVENT_LBUTTONDOWN: # captures left button double-click

        CHOSEN_POINTS[CHOSEN_POINTS_COUNTER] = x, y
        CHOSEN_POINTS_COUNTER = CHOSEN_POINTS_COUNTER + 1
        print(CHOSEN_POINTS)

def camera_calibration(imgpath,txtpath):

    global CHOSEN_POINTS_COUNTER
    global CHOSEN_POINTS
    if os.path.isfile(txtpath):
        with open(txtpath) as f:
            lines = f.read().split('\n')
            lines = lines[0:-1]#usuniecie ostatniego entera
            for line in lines:
                line = line.split(",")
                CHOSEN_POINTS[CHOSEN_POINTS_COUNTER] = line
                CHOSEN_POINTS_COUNTER=CHOSEN_POINTS_COUNTER+1
        return CHOSEN_POINTS
    cImage = cv2.imread(imgpath)
    print(CHOSEN_POINTS)

    cv2.imshow("Camera calibrator", cImage)
    cv2.setMouseCallback('Camera calibrator', click_event)
    while CHOSEN_POINTS_COUNTER < 6:
        cv2.waitKey(1)

    if CHOSEN_POINTS_COUNTER ==6:
        cv2.destroyAllWindows()
        for i in range(len(CHOSEN_POINTS)):
            print(i)
            with open(txtpath, 'a') as f:
                f.write(str(CHOSEN_POINTS[i][0])+","+str(CHOSEN_POINTS[i][1]))
                f.write("\n")
        return CHOSEN_POINTS



