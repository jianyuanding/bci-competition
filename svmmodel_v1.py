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

# 对训练数据进行预处理
def train_data_process(train_data_path):
    running_data = []
    data_path_Train = train_data_path
    with open(join(data_path_Train, f"block_1.pkl"), "rb") as infile:
        mdata1 = pickle.load(infile)

    with open(join(data_path_Train, f"block_2.pkl"), "rb") as infile:
        mdata2 = pickle.load(infile)

    with open(join(data_path_Train, f"block_3.pkl"), "rb") as infile:
        mdata3 = pickle.load(infile)

    data1 = mdata1['data']
    data2 = mdata2['data']
    data3 = mdata3['data']

    data_train = np.hstack((data1, data2, data3))  # 水平堆叠生成数组  训练集

    trigger = data_train[-1, :]
    eeg_data = data_train[0: -1, :]
    chans = list(range(1, 60))
    chans = [i - 1 for i in chans]
    eeg_data = eeg_data[chans, :]
    print("train eeg shape", eeg_data.shape)

    # trial_stimulate_mask_trig = 240  # trial开始的标志
    # trigger_idx = np.where(trigger == trial_stimulate_mask_trig)[0]  # 找到每个trial开始的标记坐标

    trial_type_left = 201
    trial_type_right = 202
    trial_type_feet = 203

    type_idx = np.where((trigger == trial_type_left) | (trigger == trial_type_right) | (trigger == trial_type_feet))[0]

    trials = []
    classes = []

    for index in type_idx:
        class_e = trigger[index]  # 对应的肢体类别
        classes.append(class_e)

    for idx in type_idx:
        trial = eeg_data[:, idx: idx + int(cal_len)]  # 取对应通道的该肢体动作发生期间的信号数据
        trials.append(trial)

    label = np.array(classes, dtype=object).astype(int)
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

    # labels = np.reshape(labelss, (labelss.shape[0], 1))
    trials_change = np.reshape(running_data, (running_data.shape[0], running_data.shape[1] * running_data.shape[2]))

    return trials_change, label

# 对测试数据进行预处理
def test_data_process(test_data_path):
    data_path_Test = test_data_path
    running_data = []

    with open(join(data_path_Test, f"block_4.pkl"), "rb") as infile:
        mdata4 = pickle.load(infile)

    with open(join(data_path_Test, f"block_5.pkl"), "rb") as infile:
        mdata5 = pickle.load(infile)

    data4 = mdata4['data']
    data5 = mdata5['data']

    data_test = np.hstack((data4, data5))  # 测试集

    trigger = data_test[-1, :]
    eeg_data = data_test[0: -1, :]
    chans = list(range(1, 60))
    chans = [i - 1 for i in chans]
    eeg_data = eeg_data[chans, :]
    print("test eeg shape", eeg_data.shape)

    # trial_stimulate_mask_trig = 240  # trial开始的标志
    # trigger_idx = np.where(trigger == trial_stimulate_mask_trig)[0]  # 找到每个trial开始的标记坐标

    trial_type_left = 201
    trial_type_right = 202
    trial_type_feet = 203

    type_idx = np.where((trigger == trial_type_left) | (trigger == trial_type_right) | (trigger == trial_type_feet))[0]

    trials = []
    classes = []

    for index in type_idx:
        class_e = trigger[index]  # 对应的肢体类别
        classes.append(class_e)

    for idx in type_idx:
        trial = eeg_data[:, idx: idx + int(cal_len)]  # 取对应通道的该肢体动作发生期间的信号数据
        trials.append(trial)

    label = np.array(classes, dtype=object).astype(int)
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

    # labels = np.reshape(labelss, (labelss.shape[0], 1))
    trials_change = np.reshape(running_data, (running_data.shape[0], running_data.shape[1] * running_data.shape[2]))

    return trials_change, label

if __name__ == '__main__':
    train_data_path = "./TrainTestData/MI/S01/Train"  # 训练集路径
    test_data_path = "./TrainTestData/MI/S01/Test"    # 测试集路径

    train_data, train_label = train_data_process(train_data_path)  # 训练数据和训练标签
    test_data, test_label = test_data_process(test_data_path)      # 测试数据和测试标签

    print("\n模型开始训练！")
    clf = svm.SVC(C=10, gamma=0.001, probability=True)  # 设置模型
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
