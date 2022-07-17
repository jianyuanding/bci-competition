import sys
import os

import pickle
import numpy as np
from os.path import join
import pandas as pd
from sklearn import svm
import joblib
from sklearn.metrics import accuracy_score
from scipy import signal

cal_time = 4
samp_rate = 250
cal_len = cal_time * samp_rate

if __name__ == '__main__':
    data_path = "./TestData/MI/S01"
    data_alls = []
    running_data = []

    a = [1, 2, 3]
    for i in a:
        with open(join(data_path, f"block_" + str(i) + f".pkl"), 'rb') as f:
            data_all = pickle.load(f)
            data_alls.append(data_all)

    with open(join(data_path, f"block_1.pkl"), "rb") as infile:
        mdata1 = pickle.load(infile)

    with open(join(data_path, f"block_2.pkl"), "rb") as infile:
        mdata2 = pickle.load(infile)

    with open(join(data_path, f"block_3.pkl"), "rb") as infile:
        mdata3 = pickle.load(infile)

    data1 = mdata1['data']
    data2 = mdata2['data']
    data3 = mdata3['data']

    data = np.hstack((data1, data2, data3))
    print("data", data.shape)
    trigger = data[-1, :]
    eeg_data = data[0: -1, :]
    # print("eeg", eeg_data.shape)

    trial_stimulate_mask_trig = 240  # trial开始的标志
    trigger_idx = np.where(trigger == trial_stimulate_mask_trig)[0]  # 找到每个trial开始的标记坐标
    print("idx", trigger_idx[1])

    trial_type_left = 201
    trial_type_right = 202
    trial_type_feet = 203

    type_idx = np.where((trigger == trial_type_left) | (trigger == trial_type_right) | (trigger == trial_type_feet))[0]

    trials = []
    classes = []

    for index in type_idx:
        class_e = trigger[index]  # 对应的肢体类别
        classes.append(class_e)

    for idx in trigger_idx:
        trial = eeg_data[:, idx: idx + int(cal_len)]  # 取对应通道的该肢体动作发生期间的信号数据
        trials.append(trial)

    label = np.array(classes, dtype=object)
    print("label", label.shape)

    trials = np.array(trials, dtype=object)
    print("trials", trials.shape)

    samples = np.array(list(zip(trials, label)), dtype=object)  # 将样本的标签和数据统一放到一起，可以方便打乱样本的顺序

    print("sample", samples.shape)

    labelss = samples[:, -1]  # 取样本标签
    trialss = samples[:, 0]  # 取样本的数据（注意这里是取出来的格式是(90, ),而不是（90,64，1000））
    print("label", labelss.shape)
    print("trials", trialss.shape)

    for i in range(len(samples)):
        running_data.append(samples[:, 0][i])

    running_data = np.array(running_data, dtype=object)
    print("run", running_data.shape)

    labels = np.reshape(labelss, (labelss.shape[0], 1))
    trials_change = np.reshape(running_data, (running_data.shape[0], running_data.shape[1] * running_data.shape[2]))

    clf = svm.SVC(C=1000, gamma=0.001, probability=True)
    clf.fit(trials_change, label.astype('int'))

    joblib.dump(clf, 'Algorithm/model/svm.joblib')  # 保存模型

    print(clf.score(trials_change, label.astype('int')))

    # print(trials_change[1])
    test = np.reshape(trials_change[20], (1, trials_change.shape[1]))
    print(test.shape)

    cdf = joblib.load('Algorithm/model/svm.joblib')
    predict = cdf.predict(test)
    print("predict", predict)
    print("label", label[20])
