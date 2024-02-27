import numpy as np
import cv2
import os

char_width = 35
char_height = 50

def recognition_by_path(model, chartest):
    label_str ="0 1 2 3 4 5 6 7 8 9 A B C D E F G H I J K L M N O P Q R S T U V W X Y Z"
    img_label = np.array(label_str.split())
    listChar = []
    for i in range(len(chartest)):
        image_resize = cv2.resize(chartest[i], (char_width, char_height)) #w, h
        cv2.imwrite(f"image_cropped\Char\img{i}.jpg",image_resize)
        prep = model.predict(image_resize.reshape(1,char_height,char_width,1))
        predicted_class = np.argmax(prep)
        predicted_label = img_label[predicted_class]
        listChar.append(predicted_label)
    return listChar

def process_license_plate(listChar):
    if len(listChar) < 8  or len(listChar) > 9:
        return []
    replace_char = {
    "Q": "0",
    "D": "0",
    "Z": "2",
    "L": "4",
    "G": "6",
    "S": "6",
    "T": "7",
    "B": "8",
    "H": "8"
    }
    part1 = ''.join(listChar[:2])  
    part2 = ''.join(listChar[2])
    part3 = ''.join(listChar[3:]) 
    part3 = part3[0] + ' ' + part3[1:]

    for key, value in replace_char.items():
        part1 = part1.replace(key, value)
        part3 = part3.replace(key, value)

    listChar = part1 + "-" + part2 + part3
    return listChar

def ReadLP(model, chartest):
    return process_license_plate(recognition_by_path(model, chartest))


# my_list = ['9', '7', 'L', '1', '9', '6', 'Z', 'B', '7']

# res = process_license_plate(my_list)
# print(type(res))
