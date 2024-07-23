import os
import json
from copy import deepcopy
import random
import numpy as np
import torch
from data_handler import GenericDataset
import os
import pickle
from util_files.ood_metrics import *
# 获取当前工作目录的绝对路径
current_working_directory = os.getcwd()
project_path = current_working_directory
import datetime
from utils import check_log_dir, make_log_name, set_seed
# 获取当前日期
current_date = datetime.date.today()

# 检查数据是否与文件中的相同的函数
def is_data_same(file_path):
    if not os.path.exists(file_path):
        return False
    return True

# 将数据保存到文件的函数
def save_data(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)

#将数据load出来
def load_data(file_path):
    with open(file_path, 'r') as file:
        file_data = json.load(file)
        return file_data

