import pickle

import matplotlib.pyplot as plt
import numpy as np
from os.path import join
from scipy import signal
from scipy.signal import butter,lfilter,filtfilt


cal_time = 4  # 时间
samp_rate = 250  # 采样率
cal_len = cal_time * samp_rate  # 样本长度

# 标准化脑电数据
def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma

# S01 (8, 30); S02 (8, 40); S03 (8, 30); S04 (8, 30); S05 (8, 30)

def band_Filter(rawdata):
    order, low_cut, high_cut, fs = 5, 8, 30, 1000
    b, a = butter(order, [low_cut, high_cut], 'bandpass', fs=fs)
    rawdata_filter = filtfilt(b, a, rawdata)
    return rawdata_filter

# 对训练数据进行预处理
def train_data_process(train_data_path):
    data_path_train = train_data_path
    # running_data_train = []

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
    chans = [24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
    # chans = np.arange(59)

    eeg_data_train = eeg_data_train[chans, :]
    eeg_data_train1 = eeg_data_train.copy()
    # print("train eeg shape", eeg_data_train.shape)

    eeg_data_train_filter = band_Filter(eeg_data_train1)

    # plt.figure(figsize=(16, 4))
    # x = eeg_data_train[1, :]
    # plt.plot(x[0:2000], label='before')
    # y = eeg_data_train_filter[1, :]
    # plt.plot(y[0:2000], label='after')
    # plt.show()
    # plt.legend()
    # # 画fft图
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文
    # plt.rcParams['axes.unicode_minus'] = False  # 显示负号
    # xf = np.fft.fft(y)  # 对离散数据y做fft变换得到变换之后的数据xf
    # xfp = np.fft.fftfreq(1000, d=1 / 250)  # fftfreq(length，fs)，计算得到频率
    # xf = abs(xf)  # 将复数求模，得到fft的幅值
    # ax1 = plt.subplot(1, 2, 2)
    # plt.plot(xfp, xf)  # 画fft图像，横坐标为频率，纵坐标为幅值
    # plt.xlabel('f(Hz)')
    # plt.ylabel('Amp')
    # plt.title('fft')
    # plt.show()

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

    # 数据增强, 相隔XX个采样点获取一次数据，取5组数据，即同一个样本扩充多组数据，1->6
    # for idx_train in type_idx_train:
        # trial_train = eeg_data_train_filter[:, idx_train: idx_train + int(cal_len)]
        # trial_train1 = eeg_data_train_filter[:, idx_train + 5: idx_train + 5 + int(cal_len)]
        # trial_train2 = eeg_data_train_filter[:, idx_train + 25: idx_train + 25 + int(cal_len)]
        # trial_train3 = eeg_data_train_filter[:, idx_train + 40: idx_train + 40 + int(cal_len)]
        # trial_train4 = eeg_data_train_filter[:, idx_train + 45: idx_train + 45 + int(cal_len)]
        # trial_train5 = eeg_data_train_filter[:, idx_train + 50: idx_train + 50 + int(cal_len)]
        # trials_train.append(trial_train)
        # trials_train.append(trial_train1)
        # trials_train.append(trial_train2)
        # trials_train.append(trial_train3)
        # trials_train.append(trial_train4)
        # trials_train.append(trial_train5)

    Train_data = np.array(trials_train).astype('float32')

    label_train = np.array(classes_train, dtype=object).astype(int)
    # print("train label shape", label_train.shape)

    # 每个标签复制六份，再转为onehot形式
    label_train_copy = label_train.reshape(-1)-201  # 无数据增强
    # label_train_copy = np.tile(label_train.reshape(-1, 1), (1, 6)).reshape(-1) - 201  # 有数据增强，复制
    Train_label = np.eye(3)[label_train_copy]  # 转为one-hot形式

    # 样本增强
    # 对样本的数值直接取相反值???
    # Train_data = np.stack((Train_data, Train_data[:,::-1,:])).reshape(1080, 20, 1000)
    # Train_label = np.stack((Train_label, Train_label)).reshape(1080,3)

    # 将输入样本翻转，即所有通道倒序插入原样本，使样本数翻倍
    # Train_data = np.stack((Train_data, Train_data[:,::-1,:])).reshape(1080, 59, 1000)
    # Train_label = np.stack((Train_label, Train_label)).reshape(1080, 3)

    # 打乱样本顺序
    shuffle_idx = np.random.permutation(Train_label.shape[0])
    train_data_shuffle, train_label_shuffle = Train_data[shuffle_idx], Train_label[shuffle_idx]

    return train_data_shuffle, train_label_shuffle

# 对测试数据进行预处理
def test_data_process(test_data_path):
    data_path_test = test_data_path
    # running_data_test = []

    with open(join(data_path_test, f"block_4.pkl"), "rb") as infile:
        mdata4 = pickle.load(infile)

    with open(join(data_path_test, f"block_5.pkl"), "rb") as infile:
        mdata5 = pickle.load(infile)

    data4 = mdata4['data']
    data5 = mdata5['data']

    data_test = np.hstack((data4, data5))  # 测试集

    trigger_test = data_test[-1, :]
    eeg_data_test = data_test[0: -1, :]
    chans = [24, 25, 26, 27, 28, 29, 30, 33, 34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47]
    # chans = np.arange(59)
    eeg_data_test = eeg_data_test[chans, :]
    eeg_data_test1 = eeg_data_test.copy()
    # print("test eeg shape", eeg_data_test.shape)

    eeg_data_test_filter = band_Filter(eeg_data_test1)

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
    # print("test label shape", label_test.shape)

    trials_test = np.array(trials_test, dtype=object)
    # print("test trials shape", trials_test.shape)

    Test_data = np.array(trials_test).astype('float32')
    label_test_copy = label_test.reshape(-1)-201
    Test_label = np.eye(3)[label_test_copy]

    shuffle_idx = np.random.permutation(Test_label.shape[0])
    test_data_shuffle, test_label_shuffle = Test_data[shuffle_idx], Test_label[shuffle_idx]

    return test_data_shuffle, test_label_shuffle