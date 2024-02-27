# Helmet-Violation-Detection-Using-YOLO-and-VGG16

## Description
This project is utilized for the purpose of detecting violations of motorbike riders not wearing helmets when participating in traffic. There are three main models used in this project: the Moto_Detect_Model (model to detect motorcycles in images), the Helmet_LP_Detect_Model (model to detect helmets and license plates), and the ReadLP_Model (model to read license plates). We have integrated these three models together according to the logic flow in the overview diagram below and developed an interface allowing direct detection of helmet violation on video.

## Overview and Explanation
![image](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/7447a70e-7597-4419-9965-75cc7a0f9988)


- **Model 1: Detect Motobike using YOLOv8**
  + **Input:** Traffic images include various vehicles.
  + **Output:** Bounding boxes for motorcycles.
  + **Implementation Approach:**
  For this first model, our dataset was collected using personal cameras, with the direction of filming aligned with the movement of the vehicles to capture videos and cut them into frames at a certain ratio to obtain a traffic image set.Subsequently, we utilized Roboflow to label and save it as a complete training dataset for this motorcycle detection model.
![image](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/ea8421df-d1ee-4a6b-ac34-ffa496186c6b)
After training with YOLOv8, the model could detect motorcycles very well. However, in certain images, some motorcycles were too far away, making the license plate illegible, which could affect the license plate reading results later. Therefore, we applied a screen threshold to control motorcycles that are too distant.
![image53](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/11b08a0e-f81f-45c2-864a-83d0b3dec157)

- **Model 2: Detect Helmet and LP using YOLOv8**
  + **Input:** Images of motorcycles.
  + **Output:** Bounding boxes for Helmets, no-helmet, and license plates.
  + **Implementation Approach:**
 After the first model successfully detected motorcycles in the original images, we relied on the coordinates of the bounding boxes to extract those motorcycles. Meanwhile, the dataset for helmets and no-helmet instances is quite scarce. Therefore, I tried to gather as many datasets on back-heads and mix them together to create this helmet dataset. I hope it works effectively! :))))
![image](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/3fc2575e-dbe0-4c59-8950-26136d464a37)
From there, we compiled a dataset of motorcycles for the second model. Subsequently, we continued to use Roboflow to label and train the model to recognize motorcycles and license plates.
![image](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/1c5e4609-2b12-4076-a44c-3307d33309eb)

- **Model 3: Read LP using VGG16(OCR Process)**
  + **Input:** Motorcycle license plate.
  + **Output:** Content of the motorcycle license plate.
  + **Implementation Approach:**
The dataset for this third model is a character dataset we collected from the internet, comprising characters from A to Z and 0 to 9. After some preprocessing steps, I used VGG16 to train this dataset.
![image](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/91c45bf1-251b-4fe5-95f2-0c954f649596)

Meanwhile, the motorcycle license plate will be cropped based on the bounding box drawn by Model 2. Then, preprocessing steps in the OCR process will be carried out, which involve numerous different steps—you can explore OCR tasks to understand more details. After contrast enhancement, noise reduction, etc., we will draw contours around the characters in the license plate and cut them out to save as a sequence of images. VGG16 will be used to recognize these cropped characters and output the content of the license plate.
![image](https://github.com/ThanhSan97/Helmet-Violation-Detection-Using-YOLO-and-VGG16/assets/91296937/ecc37b89-0e91-483b-b92f-d176edfa105e)

## Participants
1. NGUYEN DINH THANH SAN
- Major: Artificial Intelligence
- Contact:
   + Linkedin: https://www.linkedin.com/in/thanh-san-a3b45b275
   + Github: ThanhSan97
   + Gmail: sannguyen0907@gmail.com - nguyendthanhsan@gmail.com
2. NGUYEN HUYNH CHI KHANG
- Major: Artificial Intelligence
- Contact:
   + Linkedin: linkedin.com/in/nguyen-huynh-chi-khang-607a3926a
   + Github: Khang1405
   + Gmail: chikhang1235202@gmail.com
3. NGUYEN PHAN DUC THANH
- Major: Artificial Intelligence
- Contact:
   + Github: NguyenPhanDucThanh
   + Gmail: thanhnguyen1802dn@gmail.com.

## Project and Dataset Information
- **Traffic Dataset: https://universe.roboflow.com/cdio-zmfmj/motobike-detection**
- **Helmet and LP Dataset: https://universe.roboflow.com/cdio-zmfmj/helmet-lincense-plate-detection-gevlq**
- **LP Character Dataset: https://drive.google.com/file/d/18Sm22tq9vaTEtNcm8hKYxYcLokuoHRsI/view?usp=drive_link**
