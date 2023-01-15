import argparse
import math
import os, sys
import shutil
import time
import keyboard
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import scipy.special
import numpy as np
import torchvision.transforms as transforms
import PIL.Image as image
from matplotlib import pyplot as plt
from tools.PointsChoosing import camera_calibration, find_optic_middle
from tools.ImageWrapping import warp_image_to_birdseye_view, warp_point, get_warp_perspective, calculate_distance_between_points, estimate_real_distance

from lib.config import cfg
from lib.config import update_config
from lib.utils.utils import create_logger, select_device, time_synchronized
from lib.models import get_net
from lib.dataset import LoadImages, LoadStreams
from lib.core.general import non_max_suppression, scale_coords
from lib.utils import plot_one_box,show_seg_result
from lib.core.function import AverageMeter
from lib.core.postprocess import morphological_process, connect_lane
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

print(sys.path)

divide = 40
points_density = 20  # dobre 20/40
first_phase = points_density//3
second_phase = first_phase+1

normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

def average_points(unique_cars):
    unique_cars_average = []
    for car in unique_cars:
        sum_y1=0
        sum_x1=0
        for point in car:
             sum_y1 = sum_y1+point[1]
             sum_x1 = sum_x1+point[0]
        avg_y1 = sum_y1//len(car)
        avg_x1 = sum_x1//len(car)
        unique_cars_average.append((avg_x1,avg_y1))
    return unique_cars_average

def label_cars(set_of_found_cars, height):
    unique_cars = []
    number_of_frames = 6
    if len(set_of_found_cars)>number_of_frames:
        first_set = set_of_found_cars[-(number_of_frames+1)]
        for car_point in first_set:
            unique_car = []
            unique_car.append(car_point)
            for set in set_of_found_cars[-number_of_frames:-1]:
                get_point = True
                for car2 in set:
                    if abs(car2[0]-car_point[0])<height//60 and abs(car2[1]-car_point[1])<height//30 and get_point:
                        unique_car.append(car2)
                        car_point = car2
                        get_point=False
                if len(unique_car)==number_of_frames:
                    unique_cars.append(unique_car)
    return unique_cars


def separate_points(points, left_left_lane_points,left_lane_points, right_lane_points, right_right_lane_points, img_middle):
    left_distance_list = []
    right_distance_list =[]
    for point in points:
        distance = point[0] - img_middle
        if distance < 0:
            left_distance_list.append(abs(distance))
        if distance >= 0:
            right_distance_list.append(abs(distance))
    if len(left_distance_list)>0:
        min_left_idx = (left_distance_list.index((min(left_distance_list))))
        left_lane_points.append(points[min_left_idx])
        if len(left_distance_list)>1:
            left_left_lane_points.append(points[min_left_idx-1])

    if len(right_distance_list) > 0:
        min_right_idx = (right_distance_list.index(min(right_distance_list)))
        min_right_idx = min_right_idx + len(left_distance_list)
        right_lane_points.append(points[min_right_idx])
        if len(right_distance_list)>1:
            right_right_lane_points.append(points[min_right_idx+1])

    return left_left_lane_points, left_lane_points, right_lane_points, right_right_lane_points


def find_middle_pixel_on_height(lane_mask, height):
    horizontal_lane=(lane_mask[height])
    cnt0=0
    cnt1=0
    previous_pixel = 0
    points_list=[]
    for current_pixel in range(len(horizontal_lane)):
        #print(current_pixel)
        if horizontal_lane[previous_pixel]*horizontal_lane[current_pixel]==1:
            cnt1=cnt1+1
        elif horizontal_lane[previous_pixel]*horizontal_lane[current_pixel]==0 and cnt1!=0:
            points_list.append((current_pixel-cnt1//2,height))
            cnt1=0
        elif horizontal_lane[current_pixel]==0:
            cnt0=cnt0+1
        previous_pixel = current_pixel

    return points_list


def display_from_list(img, list_of_points, height, color):
    previous_element = []
    for element in list_of_points:
        # cv2.circle(img, element, 1, [0, 255, 255], 2)
        if len(previous_element) != 0:
            if abs(previous_element[0] - element[0]) < height // 5: # ll_seg_mask
                cv2.line(img, previous_element, element, color, 2)
        previous_element = element
    return img


def find_point_on_height(list_of_points, height):
    for point in list_of_points:
        if point[1] == height:
            return point


def average_point(list_of_points): # ew. wywalać tee liste, która ma [(1049, 737), None, None, None, None] tylko jeden/dwa? elementy
    average_value = 0
    i = 0
    for point in list_of_points:
        if point != None:
            i = i+1
            average_value = average_value + point[0]
    average_value = average_value/i
    wanted_point = (int(average_value), list_of_points[0][1])
    return wanted_point


def average_line(set_of_lines):
    b = 0
    c = 0
    work_list = []
    final_list = []
    for i in range(5):
        work_list.append(set_of_lines[-i - 1])
    for i in range(5):
        a = len(work_list[i])
        if a > b:
            b = a
            c = i
    longest_list = work_list[c]
    work_list.pop(c)
    for point in longest_list:
        all_points_on_height = [point]
        for other_list in work_list:
            all_points_on_height.append(find_point_on_height(other_list, point[1]))
        final_list.append(average_point(all_points_on_height))
    return final_list


def display_from_set(img, set_of_lines, mask):
    if len(set_of_lines) < 5:
        return img
    else:
        final_list = average_line(set_of_lines)
        img = display_from_list(img, final_list, mask, [0, 255, 0])
        return img


def check_points(lane_points, point, last_points, index, i):
    if len(lane_points) > 0 and point[1] == lane_points[-1][1]:
        a = math.sqrt(
            ((point[0] - lane_points[-2][0]) * (point[0] - lane_points[-2][0])) + (
                    (point[1] - lane_points[-2][1]) * (
                    point[1] - lane_points[-2][1])))
        b = math.sqrt(
            ((lane_points[-1][0] - lane_points[-2][0]) * (
                    lane_points[-1][0] - lane_points[-2][0])) + (
                    (lane_points[-1][1] - lane_points[-2][1]) * (
                    lane_points[-1][1] - lane_points[-2][1])))
        if b > a:
            lane_points.pop(-1)
            lane_points.append(point)
    elif len(lane_points) > 0 and last_points[index] == lane_points[-1]:
        lane_points.append(point)
    elif len(lane_points) > 4 and last_points[index][1] == lane_points[-1][1] and i < 15:
        a, b, c, d = 0, 0, 0, 0
        a = lane_points[-3][0]-lane_points[-4][0]
        b = lane_points[-2][0] - lane_points[-3][0]
        c = lane_points[-1][0] - lane_points[-2][0]
        d = last_points[index][0] - lane_points[-2][0]
        if a > 0 and b > 0:
            diff1 = abs(c-(a+b)/2)
            diff2 = abs(d-(a+b)/2)
            if diff1 > diff2:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
        elif a < 0 and b < 0:
            diff1 = abs(c-(a+b)/2)
            diff2 = abs(d-(a+b)/2)
            if diff1 > diff2:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
        elif a > 0 and b < 0:
            if last_points[index][0] < lane_points[-2][0] and abs(last_points[index][0]-lane_points[-2][0])<threshold:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
        elif a < 0 and b > 0:
            if last_points[index][0] > lane_points[-2][0] and abs(last_points[index][0]-lane_points[-2][0])<threshold:
                lane_points.pop(-1)
                lane_points.append(last_points[index])
                lane_points.append(point)
    return lane_points


def finding_closest_point_by_width_and_height(point, last_points, howfar, right_lane_points, left_lane_points, i):
    index = 0
    for otherpoint in last_points:  # znajdowanie najbliższego punktu z linii niżej dla danego point na wysokosci powyżej 5 linii
        howclose = math.sqrt(((point[0] - otherpoint[0]) * (point[0] - otherpoint[0])) + (
                    (point[1] - otherpoint[1]) * (point[1] - otherpoint[1])))
        # print("odleglosc punktu ", points.index(point)+1," od punktu ", last_points.index(otherpoint)+1, "=sqrt((", point[0], "-", otherpoint[0], ")^2 + (", point[1], "-",otherpoint[1], ")^2)=",howclose)
        if howclose < howfar and howclose < threshold:
            howfar = howclose
            index = last_points.index(otherpoint)  # punkt znaleziony i jego index
    if howfar != 100000000000:
        right_lane_points = check_points(right_lane_points, point, last_points, index, i)
        # if len(right_lane_points) > 0 and last_points[index] == right_lane_points[-1]:
        #     right_lane_points.append(point)
        left_lane_points = check_points(left_lane_points, point, last_points, index, i)

    return right_lane_points, left_lane_points


def checking_points_distance(list_of_points):
    for point in list_of_points:
        if point != list_of_points[-1]:
            if abs(point[0] - list_of_points[list_of_points.index(point) + 1][0]) > width_threshold:
                list_of_points.pop(list_of_points.index(point) + 1)
                checking_points_distance(list_of_points)
    return list_of_points


def deleting_far_points_from_list(list_of_points):
    inverted_list = list_of_points[::-1]
    inverted_list = checking_points_distance(inverted_list)
    list_of_points = inverted_list[::-1]
    return list_of_points


def appending_list_if_found_or_not(side_points_list, previous_lines_list):
    if len(side_points_list) > 6:
        previous_lines_list.append(side_points_list)
    if len(side_points_list) <= 6 and len(previous_lines_list) > 0:
        side_points_list = previous_lines_list[-1]
        previous_lines_list.append(side_points_list)
    return previous_lines_list


def furthest_points(line):
    xx = []
    for point in line:
        xx.append(point[0])
    max_left, max_right = min(xx), max(xx)
    return [max_left, max_right]


def aproximate_line(set_of_lines, degree, fleft, fright, horizon1, horizon2):
    iksy, igreki, poly_line = [], [], []
    if len(set_of_lines) > 5:
        avg_line = average_line(set_of_lines)
        for point in avg_line:
            iksy.append(point[0])
            igreki.append(point[1])
        polynomial = np.poly1d(np.polyfit(iksy, igreki, degree))
        a = furthest_points(avg_line)
        if fleft > a[0]:
            fleft = a[0]
        if fright < a[1]:
            fright = a[1]
        ttt = np.linspace(fleft, fright, fright - fleft + 1)  # szerokośc czyli najdalej lewa i prawa
        for x in ttt:
            if horizon2 >= int(polynomial(x)) >= horizon1:
                poly_line.append([int(x), int(polynomial(x))]) # warunek upper bottom horyzont
    return poly_line, fleft, fright


def detect(cfg,opt,calibration_points):

    logger, _, _ = create_logger(
        cfg, cfg.LOG_DIR, 'demo')

    device = select_device(logger,opt.device)
    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = get_net(cfg)
    checkpoint = torch.load(opt.weights, map_location= device)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    if half:
        model.half()  # to FP16

    # Set Dataloader
    if opt.source.isnumeric():
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()

    vid_path, vid_writer = None, None
    img = torch.zeros((1, 3, opt.img_size, opt.img_size), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    model.eval()

    inf_time = AverageMeter()
    nms_time = AverageMeter()

    # Getting image/video resolution
    dataset.__iter__()
    (path, img, img_det, vid_cap, shapes) = dataset.__next__()
    height, width, _ = img_det.shape

    global width_threshold, threshold
    width_threshold = width // divide
    threshold = math.sqrt(width*height)//divide

    # M, Minv = get_warp_perspective(calibration_points, (height, width))
    # print(type(img_det[0][0][0]))

    if calibration_points is not None:
        bottom_horizon = calibration_points[4]
        upper_horizon = calibration_points[5]  # gorny horyzont - pikselowo mniejsza wartość!
        D = bottom_horizon[1] - upper_horizon[1]
        optic_middle_bottom, optic_middle_upper = find_optic_middle(calibration_points)
        optic_middle=(((optic_middle_bottom[0]+optic_middle_upper[0])//2),((optic_middle_bottom[1]+optic_middle_upper[1])//2))
        vehicle_front = (optic_middle[0],bottom_horizon[1])
        img_middle = optic_middle[0]
        number_of_segments = calibration_points[6][0]

        y_conv = int(height / 60)  # height/60 to 1m dla 720p: 12pikseli po y = 1m
        x_conv = int(height / 15)  # height/15 to 1 m, dla 720p 3piksele po x = 1m
        M, Minv = get_warp_perspective(calibration_points, x_conv, y_conv,number_of_segments)
        # optic_middle_upper_warp = warp_point(optic_middle_upper, M)
    else:
        img_middle = width//2
        bottom_horizon = height
        upper_horizon = height//2  # gorny horyzont - pikselowo mniejsza wartość!


    set_of_lines_right = []
    set_of_lines_left = []
    set_of_found_cars = []

    fleft1, fright1 = width*2, 0
    fleft2, fright2 = width*2, 0

    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset),total = len(dataset)):

        img = transform(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        det_out, da_seg_out,ll_seg_out = model(img)
        t2 = time_synchronized()
        inf_out, _ = det_out
        inf_time.update(t2-t1, img.size(0))

        # Apply NMS
        t3 = time_synchronized()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None, agnostic=False)
        t4 = time_synchronized()

        nms_time.update(t4-t3,img.size(0))
        det = det_pred[0]

        save_path = str(opt.save_dir +'/'+ Path(path).name) if dataset.mode != 'stream' else str(opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h,w,_ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # DRIVING AREA PREDICT
        # da_predict = da_seg_out[:, :, pad_h:(height-pad_h),pad_w:(width-pad_w)]
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        # _, da_seg_mask = torch.max(da_seg_mask, 1)
        # da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        ll_predict = ll_seg_out[:, :,pad_h:(height-pad_h),pad_w:(width-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        # ll_seg_mask = ll_seg_mask.to(torch.uint8)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        # ll_seg_mask = ll_seg_mask.view('uint8')[:,::4]
        # print(type(ll_seg_mask[0][0]))
        # ll_seg_mask = np.uint8(ll_seg_mask)
        # ll_seg_mask.astype('uint8')
        # Lane line post-processing

        #ll_seg_mask = process_lane_mask(ll_seg_mask)
        #ll_seg_mask = morphological_process(ll_seg_mask, kernel_size=7, func_type=cv2.MORPH_OPEN)
        #ll_seg_mask = connect_lane(ll_seg_mask)
        #color_area = np.zeros((ll_seg_mask.shape[0],ll_seg_mask.shape[1], 3), dtype=np.uint8)
        #color_area[ll_seg_mask == 1] = [255, 0, 0]

        # ############################################################TOMEK#############################################
        img_det_copy = img_det.copy()
        birds_img = warp_image_to_birdseye_view(img_det_copy, M)
        # birds_ll_seg_mask = warp_image_to_birdseye_view(ll_seg_mask, M)
        # print(birds_img.shape,' + ', birds_ll_seg_mask.shape)
        # img_det_birdseye = cv2.bitwise_and(birds_img, birds_img, mask=birds_ll_seg_mask)

        # cv2.circle(img_det, optic_middle_upper, 2, [0, 0, 255], 5)

        # upper_horizon_warped = warp_point(upper_horizon, M) # do wyszukiwania linii na birds_eye
        # bottom_horizon_warped = warp_point(bottom_horizon, M)

        points_list = []
        left_lane_points = []
        right_lane_points = []
        left_left_lane_points = []
        right_right_lane_points = []
        last_points = []

        for i in range(points_density):
            horizontal_line = bottom_horizon[1] - (i * D // points_density)
            # horizontal_line = ll_seg_mask.shape[0]-1-(i*ll_seg_mask.shape[0]//2//points_density)
            # cv2.line(img_det, (0,horizontal_line), (ll_seg_mask.shape[1],horizontal_line),[0,0,100],1)
            points = find_middle_pixel_on_height(ll_seg_mask, horizontal_line)
            if i <= first_phase:
                left_left_lane_points, left_lane_points, right_lane_points, right_right_lane_points = separate_points(points, left_left_lane_points, left_lane_points, right_lane_points, right_right_lane_points,ll_seg_mask.shape[1]//2)
                left_lane_points = deleting_far_points_from_list(left_lane_points)
                right_lane_points = deleting_far_points_from_list(right_lane_points)

            for point in points:
                # cv2.circle(img_det, point, 1, [255, 255, 255], 2)
                howfar = 100000000000
                points_list.append(point)
                if i >= second_phase:
                    right_lane_points, left_lane_points = finding_closest_point_by_width_and_height(point, last_points, howfar, right_lane_points, left_lane_points, i)

            last_points = points

        set_of_lines_left = appending_list_if_found_or_not(left_lane_points, set_of_lines_left)
        set_of_lines_right = appending_list_if_found_or_not(right_lane_points, set_of_lines_right)

        left_line, fleft1, fright1 = aproximate_line(set_of_lines_left, 2, fleft1, fright1, upper_horizon[1], bottom_horizon[1])
        right_line, fleft2, fright2 = aproximate_line(set_of_lines_right, 2, fleft2, fright2, upper_horizon[1], bottom_horizon[1])
        img_det = display_from_list(img_det, left_line, h, (0, 255, 255))
        img_det = display_from_list(img_det, right_line, h, (0, 255, 255))
        for pointl in left_lane_points:
            cv2.circle(img_det, pointl, 2, [0, 0, 255], 6)
        for pointr in right_lane_points:
            cv2.circle(img_det, pointr, 2, [0, 0, 255], 6)

        # img_det = display_from_set(img_det, set_of_lines_right, ll_seg_mask)
        # img_det = display_from_set(img_det, set_of_lines_left, ll_seg_mask)

        img_det = img_det.astype(np.uint8)
        # img_det = cv2.resize(img_det, (1280, 720), interpolation=cv2.INTER_LINEAR)
        # img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        found_cars_points_warped = []
        found_cars_points =[]
        xyxy_list = []
        if len(det):
            det[:,:4] = scale_coords(img.shape[2:],det[:,:4],img_det.shape).round()
            for *xyxy,conf,cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf:.2f}'
                plot_one_box(xyxy, img_det , label=label_det_pred, color=[123,123,255], line_thickness=2)
                bottom_y = int(xyxy[3])
                mid_x = int((xyxy[0]+xyxy[2])/2)
                bottom_middle_point = (mid_x,bottom_y)
                found_cars_points_warped.append(warp_point(bottom_middle_point,M))
                found_cars_points.append(bottom_middle_point)
                #print(xyxy)
                xyxy_list.append([warp_point((xyxy[0],bottom_y),M),warp_point((xyxy[2],bottom_y),M)])
        # odleglosc od samochodu
        set_of_found_cars.append(found_cars_points)
        unique_cars = label_cars(set_of_found_cars,h)#zrobic sredni punkt samochodu z 5 klatek i porownwyac w danej klatce i kolejnej do predkosci
        average_cars_points = average_points(unique_cars)
        #polozenie na pasie
        cv2.circle(img_det, vehicle_front, 2, [0, 0, 255], 3)
        left_line_first_x=-1
        right_line_first_x=-1
        for point in left_line:
            if abs(point[1]-vehicle_front[1])<4:
                left_line_first_x=point[0]
                cv2.circle(img_det, point, 2, [255, 0, 255], 5)
                break
        for point in right_line:
            if abs(point[1]-vehicle_front[1])<4:
                right_line_first_x=point[0]
                cv2.circle(img_det, point, 2, [255, 0, 255], 5)
                break
        if left_line_first_x!=-1 and right_line_first_x!=-1:
            lines_middle_x = (left_line_first_x+right_line_first_x)//2
            lines_middle_point = (lines_middle_x, vehicle_front[1])
            cv2.circle(img_det, lines_middle_point, 2, [255, 0, 255], 10)
            dist_polozenie = calculate_distance_between_points(warp_point(vehicle_front,M),warp_point(lines_middle_point,M))
            dist_szerokosc_pasa = calculate_distance_between_points(warp_point((left_line_first_x,vehicle_front[1]),M),
                                                                    warp_point((right_line_first_x,vehicle_front[1]),M))
            real_polozenie = estimate_real_distance(dist_polozenie, x_conv, y_conv)
            real_szerokosc = estimate_real_distance(dist_szerokosc_pasa, x_conv, y_conv)

            if real_szerokosc[0]<2 or real_szerokosc[0] >4:
                cv2.putText(img_det, "bledny odczyt linii!", (300,30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1, [0, 0, 255], thickness=2
                            )
            elif real_polozenie[0] >0.8:
                cv2.putText(img_det, str(round(real_polozenie[0],2))+"m od srodka, zmiana pasa na lewy", (300,30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1, [125, 246, 55], thickness=2)

            elif real_polozenie[0] < -0.8:
                cv2.putText(img_det, str(round(real_polozenie[0],2))+"m od srodka, zmiana pasa na prawy ", (300,30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1, [125, 246, 55], thickness=2)

            else:
                cv2.putText(img_det, str(round(real_polozenie[0],2))+"m od srodka", (300,30),
                            cv2.FONT_HERSHEY_DUPLEX,
                            1, [125, 246, 55], thickness=2)


        for point in average_cars_points:
            cv2.circle(birds_img, warp_point(point, M), 2, [0, 0, 255], 5)
            px_distance = calculate_distance_between_points(warp_point(point, M), warp_point(vehicle_front, M))
            real_dist = estimate_real_distance(px_distance, x_conv, y_conv)
            diagonal_distnace = math.sqrt((real_dist[0] ** 2) + (real_dist[1] ** 2))
            if real_dist[0] < 0.25 and real_dist[1] < 10:
                cv2.putText(img_det, ("!!!" + str(round(diagonal_distnace, 1)) + "m!!!"), (point[0]-10,point[1]), cv2.FONT_HERSHEY_DUPLEX,
                            1, [0, 0, 255], thickness=2)
            else:
                cv2.putText(img_det, (str(round(diagonal_distnace, 1))+"m"), (point[0]-10,point[1]), cv2.FONT_HERSHEY_DUPLEX,
                            1, [125, 246, 55], thickness=1)
        cv2.circle(birds_img, warp_point(vehicle_front,M), 2, [0, 0, 255], 5)

        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h,w,_= img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                print(h,w)
            vid_writer.write(img_det)

            if keyboard.is_pressed('q') or keyboard.is_pressed('esc'):
                vid_writer.release()
                cv2.destroyAllWindows()
                sys.exit()

        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

        cv2.imshow("lanes", img_det)
        cv2.waitKey(1)
        cv2.imshow("birdseye", birds_img)
        cv2.waitKey(1)

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))


if __name__ == '__main__':
    test_path = 'inference/vid2'
    calibrate = 1
    if calibrate == 1:
        calibration_points = camera_calibration(test_path+"/calibration.png", test_path+"/calibration.txt")
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/End-to-end.pth', help='model.pth path(s)')
    parser.add_argument('--source', type=str, default=test_path, help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.8, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-dir', type=str, default=test_path+'/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    with torch.no_grad():
        detect(cfg, opt, calibration_points)
