from ultralytics import YOLO    
import tensorflow as tf
import keras
import os
import keyboard
from roboflow import Roboflow

import _Motobike
import _LP_Helmet
import _ReadLP
import _myFunc
import _trigger
import json
import cv2
from ultralytics import YOLO
import numpy as np
import pandas as pd
#---------------------------------- INIT MODEL ----------------------------------

char_width = 35
char_height = 50    

def Init_Model():
    Detect_Moto_model = YOLO('D:\AI-project\Helmet-Violation\model\Motov10l.pt')  
    ReadChar_model = keras.models.load_model("D:\AI-project\Helmet-Violation\LP_Model\LP_Modelv13.h5")
    #Load API Roboflow
    with open(r"D:\AI-project\Helmet-Violation\Main\Source\roboflow_config.json") as f:
        config = json.load(f)
        ROBOFLOW_API_KEY = config["ROBOFLOW_API_KEY"]
        ROBOFLOW_WORKSPACE_ID = config["ROBOFLOW_WORKSPACE_ID"]
        print(ROBOFLOW_WORKSPACE_ID)
        ROBOFLOW_PROJECT_ID = config["ROBOFLOW_PROJECT_ID"]
        ROBOFLOW_VERSION_NUMBER = config["ROBOFLOW_VERSION_NUMBER"]
        ROBOFLOW_SIZE = config["ROBOFLOW_SIZE"]
        f.close()
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    workspace = rf.workspace(ROBOFLOW_WORKSPACE_ID)
    project = workspace.project(ROBOFLOW_PROJECT_ID)
    version = project.version(ROBOFLOW_VERSION_NUMBER)
    Detect_LP_model = version.model
    return Detect_Moto_model, Detect_LP_model, ReadChar_model

def Program(Detect_Moto_model, Detect_LP_model, ReadChar_model, path, start_frame = 0, stop_frame = 0):
    stop_flag = False
    cap = cv2.VideoCapture(path)
    seconds_interval = 0.7
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames_to_extract = int(fps * seconds_interval) # 30 * 1 = 30 frame_to_extract
    current_frame = 0 
    frame_count = 0  
    traffic_dir_path = r"D:\AI-project\Helmet-Violation\img\Traffic"
    moto_dir_path = r"D:\AI-project\Helmet-Violation\img\Moto"
    _myFunc.delete_files(traffic_dir_path)
    _myFunc.delete_files(moto_dir_path)
    while cap.isOpened() and not stop_flag:
        success, frame = cap.read()
        if success:
            if frame_count == 0:
                if stop_frame != 0:
                    if current_frame < stop_frame: 
                        current_frame += 1
                        frame_count = (frame_count + 1) % frames_to_extract
                        continue
                index_img = current_frame // frames_to_extract
                image_traffic_path = os.path.join(traffic_dir_path, f"image{current_frame // frames_to_extract:04d}.jpg")
                cv2.imwrite(image_traffic_path, frame)
                motobike_img = _Motobike.image_detect(Detect_Moto_model , frame, screen_threshold=0.1, index_img = index_img)
                for filename in os.listdir(moto_dir_path):
                    if filename.startswith(f"image{index_img}_"):
                        motobike_path = os.path.join(moto_dir_path, filename)
                        print(motobike_path)
                        if motobike_path:
                            chartest, plate_img, helmet_img, nohelmet_img = _LP_Helmet.image_detect(Detect_LP_model, motobike_path)
                            if np.array(chartest).any() and np.array(plate_img).any() and (np.array(helmet_img).any() or np.array(nohelmet_img)):
                                if(len(chartest) > 0):
                                    lp_infor = _ReadLP.ReadLP(ReadChar_model, chartest)
                                    name, gmail = _myFunc.get_client_info(lp_infor)
                                    print(name, gmail, lp_infor)
                                else:
                                    print("Error to Read LP") 
                            else:
                                continue
                        else:
                            continue
            if keyboard.is_pressed("q"):
                stop_flag = True
            current_frame += 1
            frame_count = (frame_count + 1) % frames_to_extract
        else:
            break
    cap.release()
    return current_frame

# # -------------------------------------------------------- TEST --------------------------------------------------------
