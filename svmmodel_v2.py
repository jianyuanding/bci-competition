import sys
import os
import pickle
import numpy as np
from os.path import join
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from scipy import signal

cal_time = 4
samp_rate = 250
cal_len = cal_time * samp_rate


def band_Filter(eeg_data):
    bandVue = [8, 30]
    b, a = signal.butter(6, [2 * bandVue[0] / 250, 2 * bandVue[1] / 250], 'bandpass',
                         analog=True)
    for iChan in range(eeg_data.shape[0]):
        eeg_data[iChan, :] = signal.filtfilt(b, a, eeg_data[iChan, :])
    return eeg_data

# 对训练数据进行预处理
def train_data_process(train_data_path):
    running_data_train = []
    data_path_train = train_data_path
    with open(join(data_path_train, f"block_1.pkl"), "rb") as infile:
        mdata1 = pickle.load(infile)

    with open(join(data_path_train, f"block_2.pkl"), "rb") as infile:
        mdata2 = pickle.load(infile)

    with open(join(data_path_train, f"block_3.pkl"), "rb") as infile:
        mdata3 = pickle.load(infile)

    data1 = mdata1['data']
    data2 = mdata2['data']
    data3 = mdata3['data']

    data_train = np.hstack((data1, data2, data3))  # 水平堆叠生成数组  训练集

    trigger_train = data_train[-1, :]
    eeg_data_train = data_train[0: -1, :]
    chans = list(range(1, 60))
    chans = [i - 1 for i in chans]
    eeg_data_train = eeg_data_train[chans, :]
    eeg_data_train1 = eeg_data_train.copy()
    print("train eeg shape", eeg_data_train.shape)

    eeg_data_train_filter = band_Filter(eeg_data_train1)

    # trial_stimulate_mask_trig = 240  # trial开始的标志
    # trigger_idx = np.where(trigger == trial_stimulate_mask_trig)[0]  # 找到每个trial开始的标记坐标

    trial_type_left = 201
    trial_type_right = 202
    trial_type_feet = 203

    type_idx_train = np.where((trigger_train == trial_type_left) | (trigger_train == trial_type_right) | (trigger_train == trial_type_feet))[0]

    trials_train = []
    classes_train = []

    for index_train in type_idx_train:
        class_e = trigger_train[index_train]  # 对应的肢体类别
        classes_train.append(class_e)

    for idx_train in type_idx_train:
        trial_train = eeg_data_train_filter[:, idx_train: idx_train + int(cal_len)]  # 取对应通道的该肢体动作发生期间的信号数据
        trials_train.append(trial_train)

    label_train = np.array(classes_train, dtype=object).astype(int)
    print("train label shape", label_train.shape)

    trials_train = np.array(trials_train, dtype=object)
    print("train trials shape", trials_train.shape)

    samples_train = np.array(list(zip(trials_train, label_train)), dtype=object)  # 将样本的标签和数据统一放到一起，可以方便打乱样本的顺序

    print("train sample shape", samples_train.shape)

    labelss_train = samples_train[:, -1]  # 取样本标签
    trialss = samples_train[:, 0]  # 取样本的数据（注意这里是取出来的格式是(90, ),而不是（90,64，1000））
    print("train label shape", labelss_train.shape)
    print("train trials shape", trialss.shape)

    for i in range(len(samples_train)):
        running_data_train.append(samples_train[:, 0][i])

    running_data_train = np.array(running_data_train, dtype=object)
    print("run train data shape", running_data_train.shape)

    # labels = np.reshape(labelss, (labelss.shape[0], 1))
    trials_change = np.reshape(running_data_train, (running_data_train.shape[0], running_data_train.shape[1] * running_data_train.shape[2]))

    return trials_change, label_train

# 对测试数据进行预处理
def test_data_process(test_data_path):
    data_path_test = test_data_path
    running_data_test = []

    with open(join(data_path_test, f"block_4.pkl"), "rb") as infile:
        mdata4 = pickle.load(infile)

    with open(join(data_path_test, f"block_5.pkl"), "rb") as infile:
        mdata5 = pickle.load(infile)

    data4 = mdata4['data']
    data5 = mdata5['data']

    data_test = np.hstack((data4, data5))  # 测试集

    trigger_test = data_test[-1, :]
    eeg_data_test = data_test[0: -1, :]
    chans = list(range(1, 60))
    chans = [i - 1 for i in chans]
    eeg_data_test = eeg_data_test[chans, :]
    eeg_data_test1 = eeg_data_test.copy()
    print("test eeg shape", eeg_data_test.shape)

    eeg_data_test_filter = band_Filter(eeg_data_test1)

    # trial_stimulate_mask_trig = 240  # trial开始的标志
    # trigger_idx = np.where(trigger == trial_stimulate_mask_trig)[0]  # 找到每个trial开始的标记坐标

    trial_type_left = 201
    trial_type_right = 202
    trial_type_feet = 203

    type_idx_test = np.where((trigger_test == trial_type_left) | (trigger_test == trial_type_right) | (trigger_test == trial_type_feet))[0]

    trials_test = []
    classes_test = []

    for index_test in type_idx_test:
        class_e_test = trigger_test[index_test]  # 对应的肢体类别
        classes_test.append(class_e_test)

    for idx_test in type_idx_test:
        trial_test = eeg_data_test_filter[:, idx_test: idx_test + int(cal_len)]  # 取对应通道的该肢体动作发生期间的信号数据
        trials_test.append(trial_test)

    label_test = np.array(classes_test, dtype=object).astype(int)
    print("test label shape", label_test.shape)

    trials_test = np.array(trials_test, dtype=object)
    print("test trials shape", trials_test.shape)

    samples_test = np.array(list(zip(trials_test, label_test)), dtype=object)  # 将样本的标签和数据统一放到一起，可以方便打乱样本的顺序

    print("test sample shape", samples_test.shape)

    labelss_test = samples_test[:, -1]  # 取样本标签
    trialss_test = samples_test[:, 0]  # 取样本的数据（注意这里是取出来的格式是(90, ),而不是（90,64，1000））
    print("test label shape", labelss_test.shape)
    print("test trials shape", trialss_test.shape)

    for i in range(len(samples_test)):
        running_data_test.append(samples_test[:, 0][i])

    running_data_test = np.array(running_data_test, dtype=object)
    print("run test data shape", running_data_test.shape)

    # labels = np.reshape(labelss, (labelss.shape[0], 1))
    trials_change_test = np.reshape(running_data_test, (running_data_test.shape[0], running_data_test.shape[1] * running_data_test.shape[2]))

    return trials_change_test, label_test

if __name__ == '__main__':
    train_data_path = "./TrainTestData/MI/S01/Train"  # 训练集路径
    test_data_path = "./TrainTestData/MI/S01/Test"    # 测试集路径

    train_data, train_label = train_data_process(train_data_path)  # 训练数据和训练标签
    test_data, test_label = test_data_process(test_data_path)      # 测试数据和测试标签

    print("\n模型开始训练！")
    clf = svm.SVC(C=100, gamma=0.00001, probability=True)  # 设置模型
    clf.fit(train_data, train_label)      # 训练模型
    print("\n模型训练完成！")

    joblib.dump(clf, 'svm.joblib')        # 保存训练好的模型
    print("\n保存训练好的模型参数！")

    print("\n训练准确率 =", clf.score(train_data, train_label))

    ###******  对模型进行测试  ******###
    print("\n将保存的模型放在测试集上测试")
    cdf = joblib.load('svm.joblib')       # 导入保存的模型
    predict = cdf.predict(test_data)      # 对测试集进行预测

    # 输出预测报告
    print("\n模型在测试集上的预测报告：")
    print(classification_report(test_label, predict, digits=4))

    # 计算预测准确度
    print('predict accuracy =', accuracy_score(test_label, predict))

    # 绘制混淆矩阵
    print("\n真实标签与预测值的混淆矩阵：")
    cm = confusion_matrix(test_label, predict)
    print(cm)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
