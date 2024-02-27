from ultralytics import YOLO
import cv2
import os
import shutil
import glob
import _myFunc
from PIL import Image

def image_detect(model, image, folder_result, screen_threshold = 0.2, index_img = 0):
    img = image.copy()
    results = model.predict(image, save=True, imgsz=640, conf=0.7, device = "CPU") #Detect motobike

    parent_dir = r"runs/detect"
    latest_image_path = _myFunc.get_latest_image_path(parent_dir)
    latest_img = cv2.imread(latest_image_path)
    img_width, img_height, _ = latest_img.shape
    # screen_threshold_x = int(img_width * screen_threshold)
    screen_threshold_y = int(img_height * screen_threshold)
    cv2.line(latest_img, (0, screen_threshold_y), (img_width*2, screen_threshold_y), (0, 255, 0), 2) #Draw threshold line

    if results and len(results[0]) > 0:
        for r in results:
            boxes = r.boxes.xyxy
            for idx, box in enumerate(boxes.numpy()):
                x_min, y_min, x_max, y_max = map(int, box)
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                cv2.circle(latest_img, (center_x, center_y), 5, (0, 255, 0), -1)
                motobike_path = fr'D:\AI-project\Helmet-Violation\img\Moto\{folder_result}\image{index_img}_motobike{idx}.jpg'
                if center_y >= screen_threshold_y:
                    cropped_image = image[y_min:y_max, x_min:x_max]
                    cv2.imwrite(motobike_path, cropped_image)

        cv2.imwrite(fr'D:\AI-project\Helmet-Violation\img\Thresh_result\{folder_result}\image{index_img}.jpg', latest_img) #Save threshold with center point
        return img
    else:
        return []
















def check(list, value):
    for i in range(len(list)):
        if list[i] == value:
            return True
    return False

    # list_id = []
    # img = image.copy()
    # results = model.track(image, persist=True, save=True, imgsz=640, conf=0.7, device = "CPU") #Detect motobike
    # parent_dir = r"runs/detect"
    # latest_image_path = _myFunc.get_latest_image_path(parent_dir)
    # latest_img = cv2.imread(latest_image_path)
    # img_width, img_height, _ = latest_img.shape
    # # screen_threshold_x = int(img_width * screen_threshold)
    # screen_threshold_y = int(img_height * screen_threshold)
    # cv2.line(latest_img, (0, screen_threshold_y), (img_width*2, screen_threshold_y), (0, 255, 0), 2) #Draw threshold line

    # if results and len(results[0]) > 0:
    #     for r in results:
    #         boxes = r.boxes.xyxy
    #         ids = r.boxes.id
    #         if boxes is not None and ids is not None:
    #             for box, id in zip(boxes, ids):
    #                 x_min, y_min, x_max, y_max = map(int, box)
    #                 center_x = (x_min + x_max) // 2
    #                 center_y = (y_min + y_max) // 2
    #                 cv2.circle(latest_img, (center_x, center_y), 5, (0, 255, 0), -1)
    #                 motobike_path = fr'D:\AI-project\Helmet-Violation\img\Moto\image{index_img}_motobike{id}.jpg'
    #                 if  center_y >= screen_threshold_y and check(list_id, id) == False:
    #                     list_id.append(id)
    #                     cropped_image = image[y_min:y_max, x_min:x_max]
    #                     cv2.imwrite(motobike_path, cropped_image)

    #     cv2.imwrite(fr'D:\AI-project\Helmet-Violation\img\Thresh_result\image{index_img}.jpg', latest_img) #Save threshold with center point
    #     return img
    # else:
    #     return []

def folder_detect(model, dir_path):
    for index_img, filename in enumerate(os.listdir(dir_path)):
        if filename.endswith((".jpg", ".jpeg", ".png")):
            print(f"Image_{index_img}")
            image_path = os.path.join(dir_path, filename)
            image = cv2.imread(image_path)
            image_detect(model , image, screen_threshold=0.2, index_img = index_img)

