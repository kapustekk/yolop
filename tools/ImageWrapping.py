import numpy as np
import cv2


def warp_point(point, M):
    x = point[0]
    y = point[1]
    d = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    xM = int((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / d)
    yM = int((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / d)
    return [xM, yM]


def get_warp_perspective(calibration_points, img_size):
    src, dst = get_warp_points(calibration_points, img_size[0], img_size[1])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv


def warp_image_to_birdseye_view(image, M):
    height = image.shape[0]
    width = image.shape[1]
    warped_image = cv2.warpPerspective(image, M, (height, width), flags=cv2.INTER_LINEAR)

    return warped_image


def get_warp_points(calibration_points, width, height):
    # Save corner values for source and destination partitions
    corners = np.float32([calibration_points[0],calibration_points[1],calibration_points[2],calibration_points[3]])
    # Save top left and right explicitly and offset
    offset = [50, 0]
    dst_height = 5*int(height/60) #height/60 to 1m dla 720p: 12pikseli - 1m
    dst_width = 2.5*int(height/15)
    dst0 = corners[0]
    dst1=(int(corners[0][0]+dst_width),int(corners[1][1]))
    dst2=(int(corners[0][0]+dst_width),int(corners[1][1]-dst_height))
    dst3=(int(corners[0][0]),int(corners[0][1]-dst_height))
    src_points = np.float32([corners[0], corners[1], corners[2], corners[3]])
    dst_points = np.float32([dst0,  dst1, dst2,dst3])

    return src_points, dst_points
