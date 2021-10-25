import pandas as pd
import os
import csv

def load_data(path_train, path_test):
    data_train = pd.read_csv(path_train, skipinitialspace = True)
    data_test = pd.read_csv(path_test, skipinitialspace = True)
    return data_train, data_test

def set_sepcial_observation(y_pred, special_index):
    for idx in special_index:
        y_pred[special_index] = 0
    return y_pred
        

def write_to_csv(y_pred, file_name):
    path = os.path.join("./submission/", file_name)
    with open(path, 'w', newline='') as csvf:
        # 建立 CSV 檔寫入器
        writer = csv.writer(csvf)
        writer.writerow(['id','label'])
        for i in range(int(y_pred.shape[0])):
            writer.writerow([i + 1, int(y_pred[i])])