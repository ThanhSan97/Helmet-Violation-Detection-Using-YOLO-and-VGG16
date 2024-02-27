from ultralytics import YOLO
import cv2
import numpy as np
import os
import Preprocess
import matplotlib.pyplot as plt
import math
import torch
import keras

#================================================================================
char_width = 35
char_height = 50


def find_contours(dimensions, img, check):
    w = char_width + 10
    h = char_height + 10
    char = np.array([])
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width = dimensions[0]
    upper_width = dimensions[1]
    lower_height = dimensions[2]
    upper_height = dimensions[3]
    if check == 1 : 
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:6]
    else :
        cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:5]
    ii = cv2.imread('contour.jpg')
    x_cntr_list = []
    img_res = []
    for cntr in cntrs:
        intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
        # checking the dimensions of the contour to filter out the characters by contour's size
        if intWidth > lower_width and intWidth < upper_width and intHeight > lower_height and intHeight < upper_height:
            x_cntr_list.append(intX)  # stores the x coordinate of the character's contour, to used later for indexing the contours

            char_copy = np.zeros((h,w))
            expand_x = 5
            expand_y = 3
            char = img[intY - expand_y:intY + intHeight + expand_y, intX - expand_x:intX + intWidth + expand_x]
            if char.any():
                char = cv2.resize(char, (w, h),interpolation=cv2.INTER_LINEAR)
                # char = cv2.resize(char, (w, h),interpolation=cv2.INTER_CUBIC)
                cv2.rectangle(ii, (intX,intY), (intWidth+intX, intY+intHeight), (50,21,200), 2)
                # plt.imshow(ii, cmap='gray')
                char = cv2.subtract(255, char)
                char_copy[0:h, 0:w] = char
                img_res.append(char_copy)  # List that stores the character's binary image (unsorted)
            else:
                return char
    # Return characters on ascending order with respect to the x-coordinate (most-left character first)
    # plt.show()
    # arbitrary function that stores sorted list of character indeces
    if char.any(): 
        indices = sorted(range(len(x_cntr_list)), key=lambda k: x_cntr_list[k])
        img_res_copy = []
        for idx in indices:
            img_res_copy.append(img_res[idx])  # stores character images according to their index
        img_res = np.array(img_res_copy)
        return img_res


def white_border(dilated_image):
    white = (255,255,255)
    thickness = 1
    dilated_image = cv2.line(dilated_image, (0, 0), (dilated_image.shape[1],0), color=white, thickness= thickness)
    dilated_image = cv2.line(dilated_image, (dilated_image.shape[1],0), (dilated_image.shape[1],dilated_image.shape[0]), color=white, thickness= thickness)
    dilated_image = cv2.line(dilated_image,  (dilated_image.shape[1],dilated_image.shape[0]), (0,dilated_image.shape[0]), color=white, thickness= thickness)
    dilated_image = cv2.line(dilated_image,  (0,dilated_image.shape[0]), (0, 0), color=white, thickness= thickness)
    return dilated_image

def segment_characters(image, check) :
    # Preprocess cropped license plate image
    img = cv2.resize(image, (333,75), interpolation= cv2.INTER_LINEAR)
    kerel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thre_mor = cv2.morphologyEx(img, cv2.MORPH_DILATE, kerel3)
    _, img_binary = cv2.threshold(thre_mor, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_binary = white_border(img_binary)
    LP_WIDTH = img_binary.shape[0] 
    LP_HEIGHT = img_binary.shape[1] 
    # Estimations of character contours sizes of cropped license plates
    dimensions = [LP_WIDTH/33,LP_WIDTH,LP_HEIGHT/7, LP_HEIGHT]
    cv2.imwrite('contour.jpg',img_binary)
    # Get contours within cropped license plate
    char_list = find_contours(dimensions, img_binary , check)
    # print(char_list)
    if char_list is None: 
        return np.array([])
    else:
        return char_list


def get_closer_plate(plate):
    imgGrayscaleplate, imgThreshplate = Preprocess.preprocess(plate)
    _, imgGrayscaleplate = cv2.threshold(imgGrayscaleplate, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    dilated_image = cv2.dilate(imgGrayscaleplate, (3,3))
    img1 = dilated_image
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    screenCnt = []
    for c in contours:
        peri = cv2.arcLength(c, True)  # Tính chu vi
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # làm xấp xỉ đa giác, chỉ giữ contour có 4 cạnh
        [x, y, w, h] = cv2.boundingRect(approx.copy())
        if (len(approx) == 4):
            screenCnt.append(approx)

            cv2.putText(plate, str(len(approx.copy())), (x, y), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)

        for screenCnt in screenCnt:
            cv2.drawContours(plate, [screenCnt], -1, (0, 255, 0), 3)  # Khoanh vùng biển số xe

            ############## Find the angle of the license plate #####################
            (x1, y1) = screenCnt[0, 0]
            (x2, y2) = screenCnt[1, 0]
            (x3, y3) = screenCnt[2, 0]
            (x4, y4) = screenCnt[3, 0]
            array = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            array.sort(reverse=True, key=lambda x: x[1])
            (x1, y1) = array[0]
            (x2, y2) = array[1]
            doi = abs(y1 - y2)
            ke = abs(x1 - x2)
            angle = math.atan(doi / ke) * (180.0 / math.pi)

            ####################################

            ########## Crop out the license plate and align it to the right angle ################

            mask = np.zeros(imgGrayscaleplate.shape, np.uint8)
            new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
            # cv2.imshow("new_image",new_image)
            # cv2.waitKey()
            # Cropping
            (x, y) = np.where(mask == 255)
            (topx, topy) = (np.min(x), np.min(y))
            (bottomx, bottomy) = (np.max(x), np.max(y))

            roi = plate[topx:bottomx, topy:bottomy]
            imgThresh = imgThreshplate[topx:bottomx, topy:bottomy]
            ptPlateCenter = (bottomx - topx) / 2, (bottomy - topy) / 2

            if x1 < x2:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, -angle, 1.0)
            else:
                rotationMatrix = cv2.getRotationMatrix2D(ptPlateCenter, angle, 1.0)

            roi = cv2.warpAffine(roi, rotationMatrix, (bottomy - topy, bottomx - topx))
            imgThresh = cv2.warpAffine(imgThresh, rotationMatrix, (bottomy - topy, bottomx - topx))
            img1 = imgThresh
    return img1


def image_detect(model, path):
    char = ""
    check_LP = 0 
    check_helmet = 0
    check_nohelmet = 0
    helmet_img = 1
    plate_img = 1
    results = model.predict(path,confidence=70, overlap = 50)
    results_json = results.json()
    if results_json['predictions'] == []:
        print(f"No predictions for {path} at confidence: {70} and overlap {50}")
        return np.array([]), np.array([]), np.array([]), np.array([])
    else:
        original_file = os.path.basename(path).split('/')[-1]
        results.save(fr"D:\AI-project\Helmet-Violation\img\LP\{original_file}")
    # Show the results
        im = cv2.imread(path)
        # print(results)
        for r in results:
            check = r['class']
            if check == "helmet":
                check_helmet = 1 
                x0 = r['x'] - r['width'] / 2#start_column
                x1 = r['x'] + r['width'] / 2#end_column
                y0 = r['y'] - r['height'] / 2#start row
                y1 = r['y'] + r['height'] / 2#end_row
                helmet_img1 = im[int(y0):int(y1), int(x0):int(x1)]
                helmet_img = helmet_img1
            if check == "no helmet":
                check_nohelmet = 1 
                x0 = r['x'] - r['width'] / 2#start_column
                x1 = r['x'] + r['width'] / 2#end_column
                y0 = r['y'] - r['height'] / 2#start row
                y1 = r['y'] + r['height'] / 2#end_row
                nohelmet_img1 = im[int(y0):int(y1), int(x0):int(x1)]
                nohelmet_img = nohelmet_img1
            if check == "LP":   
                check_LP = 1
                x0 = r['x'] - r['width'] / 2#start_column
                x1 = r['x'] + r['width'] / 2#end_column
                y0 = r['y'] - r['height'] / 2#start row
                y1 = r['y'] + r['height'] / 2#end_row
                plate = im[int(y0):int(y1), int(x0):int(x1)]
                plate_img = plate.copy()
                plate = get_closer_plate(plate)
                cropped_top = plate[0:int(plate.shape[0]/2) + 3, 0 : plate.shape[1]]
                cropped_under = plate[int(plate.shape[0]/2) - 3 : plate.shape[0]]
                char1 = segment_characters(cropped_top, 0)
                char2 = segment_characters(cropped_under, 1)
                char = []
                if char1.any() and char2.any():
                    char = np.concatenate((char1, char2))

        if check_LP == 1 and check_nohelmet == 1 and check_helmet == 1: #Có cả 3
            check_LP = check_nohelmet = check_helmet = 0
            return char, plate_img, helmet_img, nohelmet_img
        elif check_LP == 1 and check_nohelmet == 1 and check_helmet == 0: #Không có class helmet
            check_LP = check_nohelmet = 0
            return char, plate_img, np.array([]), nohelmet_img
        elif check_LP == 1 and check_helmet == 1 and check_nohelmet == 0: #không có class no-helmet
            check_LP = check_helmet = 0
            return char, plate_img, helmet_img, np.array([])
        else :
            return np.array([]), np.array([]), np.array([]), np.array([])
            
def detect_moto_cropped_save(dir_path):
    for filename in os.listdir(dir_path):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            image_path = os.path.join(dir_path, filename)
            image_detect(image_path)

