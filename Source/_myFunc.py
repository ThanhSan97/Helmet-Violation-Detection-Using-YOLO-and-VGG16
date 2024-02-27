import os
import pandas as pd
import csv
import random


def get_latest_image_path(parent_dir):
    all_files = []
    for root, dirs, files in os.walk(parent_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                file_path = os.path.join(root, file)
                all_files.append((file_path, os.path.getmtime(file_path)))

    if not all_files:
        return None
    latest_image_path = max(all_files, key=lambda x: x[1])[0]
    
    return latest_image_path

def fakeData(lp, file_path):
    names = ["Le Tuan Anh", "Le Thi Thao Vy", "Tran Nguyen Nhu Ngoc","Bui Thi Kha Nhu",
            "Nguyen Dinh Thanh San", "Nguyen Huynh Chi Khang", "Nguyen Phuoc Trinh", "Nguyen Phan Duc Thanh"]

    info = {
        "Nguyen Dinh Thanh San": {"gmail": "thanhsan10b10@gmail.com", "address": "Quang Tri"},
        "Nguyen Huynh Chi Khang": {"gmail": "chikhang1235202@gmail.com", "address": "Gia Lai"},
        "Nguyen Phuoc Trinh": {"gmail": "phuoctrinh2k2@gmail.com", "address": "Quang Nam"},
        "Nguyen Phan Duc Thanh": {"gmail": "thanhnguyen1802dn@gmail.com", "address": "Da Nang"},
        "Le Tuan Anh": {"gmail": "thanhsan10b10@gmail.com", "address": "Quang Tri"},
        "Le Thi Thao Vy": {"gmail": "chikhang1235202@gmail.com", "address": "Gia Lai"},
        "Tran Nguyen Nhu Ngoc": {"gmail": "phuoctrinh2k2@gmail.com", "address": "Quang Nam"},
        "Bui Thi Kha Nhu": {"gmail": "thanhnguyen1802dn@gmail.com", "address": "Da Nang"},
    }

    with open(file_path, 'a', newline='') as file:
        writer = csv.writer(file)
        with open(file_path, 'r') as check_file:
            check_reader = csv.reader(check_file)
            header = next(check_reader, None)
            if header is None or header != ["STT", "Name", "Gmail", "Address", "LP"]:
                writer.writerow(["STT", "Name", "Gmail", "Address", "LP"])

        try:
            with open(file_path, 'r') as read_file:
                reader = csv.DictReader(read_file)
                stt_values = [int(row["STT"]) for row in reader]
                count = max(stt_values) + 1
        except FileNotFoundError:
            count = 1
        name = random.choice(names)
        gmail = info[name]["gmail"]
        address = info[name]["address"]
        writer.writerow([count, name, gmail, address, lp])
        count+=1


def get_client_info(lp_info, file_path):
    data = pd.read_csv(file_path)
    try:
        row = data[data['LP'] == lp_info].iloc[0]
        ten = row['Name']
        gmail = row['Gmail']

        return ten, gmail
    except IndexError:
        return None, None

def delete_files(path):
    [os.remove(os.path.join(path, tep_tin)) for tep_tin in os.listdir(path)]

def create_folder(path):
    os.makedirs(path)

def FilePreProcess(path):
    if not os.path.exists(path):
        create_folder(path)
    else:
        delete_files(path)

# # # Sử dụng hàm
# lp_infor = "43-C1 95574"
# user_data_path = r'D:\AI-project\Helmet-Violation\User_Data_6.csv'
# ten, gmail = get_client_info(lp_infor, user_data_path)  # Thay đổi số 3 thành số lp bạn muốn truy xuất

# if ten is not None and gmail is not None:
#     print(f'Tên: {ten}, Gmail: {gmail}')
# else:
#     print('Không tìm thấy dữ liệu cho lp cần truy xuất.')