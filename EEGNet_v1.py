import numpy as np
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
# PyRiemann imports
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
import scipy.io
from matplotlib import pyplot as plt
from model import EEGNet
from Data_process import train_data_process
from Data_process import test_data_process
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import warnings
warnings.filterwarnings("ignore")


kernels, chans, samples = 1, 20, 1000

train_data_path = "E:/Knowledge/bci_competition/MI/mi_debug/TestData/MI/S05/"
# train_data_path = "./TrainTestData/MI/S01/Train"  # 训练集路径
test_data_path = "./TrainTestData/MI/S01/Test"  # 测试集路径

train_datas, train_labels = train_data_process(train_data_path)  # 训练数据和训练标签
test_datas, test_labels = test_data_process(test_data_path)

mu = np.mean(train_datas, axis=0)
sigma = np.std(train_datas, axis=0)

# //add noise，增加网络训练难度，达到一定的正则效果，减小过拟合，提高泛化能力
noise=np.random.normal(0, 1/5, [90, 20, 1000])  # 1/500
train_datas=np.add(train_datas, noise)

np.save(file="./standard/mu5.npy", arr=mu)    # 保存标准化的参数
np.save(file="./standard/sigma5.npy", arr=sigma)

train_data = np.array((train_datas-mu)/sigma).astype('float32')
test_data = np.array((test_datas-mu)/sigma).astype('float32')

# take percent of the data to train/validate/test
X_train = train_data[:60]
Y_train = train_labels[:60]
X_validate = train_data[60:80]
Y_validate = train_labels[60:80]
X_test = train_data[80:90]
Y_test = train_labels[80:90]
# X_train = train_data[:70]
# Y_train = train_labels[:70]
# X_validate = train_data[70:90]
# Y_validate = train_labels[70:90]
# X_test = test_data
# Y_test = test_labels

#根据网络结构设置数据的输入形式(trials, channels, samples, kernels)
X_train = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# 训练模型
model = EEGNet(nb_classes = 3, Chans = 20, Samples = 1000,
               dropoutRate = 0.5, kernLength = 128, F1 = 8, D = 2, F2 = 16,
               dropoutType = 'Dropout')
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])
# count number of parameters in the model
numParams = model.count_params()
# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='./Model/EEGNetcheckpoint5.h5', monitor='val_accuracy', verbose=1,
                               save_best_only=True)
class_weights = {0: 1, 1: 1, 2: 1, 3: 1}
fittedModel = model.fit(X_train, Y_train, batch_size=8, epochs=300,
                        verbose=2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight=class_weights)

model.load_weights('./Model/EEGNetcheckpoint5.h5')
probs = model.predict(X_test)
preds = probs.argmax(axis=-1)
# acc = np.mean(preds == Y_test.argmax(axis=-1))
# print("Test classification accuracy: %f " % (acc))

# 输出预测报告
print("\n模型在测试集上的预测报告：")
print(classification_report(Y_test.argmax(axis=-1), preds, digits=4))

# 计算预测准确度
print('predict accuracy =', accuracy_score(Y_test.argmax(axis=-1), preds))

# 绘制混淆矩阵
print("\n真实标签与预测值的混淆矩阵：")
cm = confusion_matrix(Y_test.argmax(axis=-1), preds)
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# plot the accuracy and loss graph
plt.plot(fittedModel.history['accuracy'])
plt.plot(fittedModel.history['val_accuracy'])
plt.plot(fittedModel.history['loss'])
plt.plot(fittedModel.history['val_loss'])
plt.title('acc & loss')
plt.xlabel('epoch')
plt.legend(['tra_acc', 'val_acc','tra_loss','val_loss'], loc='upper right')
plt.show()