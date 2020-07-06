import os
import pickle
import time 
import sys
import random 
import tensorflow as tf 
import numpy as np 
# -----------------------------
import getConfig 
from cnnModel import cnnModel

gConfig = {}
gConfig = getConfig.get_config(config_file = 'config.ini')

def read_data(dataset_path, im_dim, num_channels, num_files, images_per_file):
    files_name = os.listdir(dataset_path)
    # 创建空的二维数组用于存放图像二进制信息
    dataset_array = np.zeros(shape = (num_files * images_per_file, im_dim, im_dim, num_channels))
    # 创建空数组存放图像的标注信息
    dataset_labels = np.zeros(shape = (num_files * images_per_file))

    index = 0
    # 从训练集中读取二进制数据并将其维度转换成32*32*3
    for file_name in files_name:
        if file_name[0: len(file_name)-1] == 'data_batch_':
            print('正在处理数据：', file_name)
            # 这一步以后应该会很常用
            data_dict = unpickle_patch(dataset_path + file_name)
            images_data = data_dict[b'data']
            print('转化前格式：',images_data.shape)

            # 将格式转换为32*32*3
            # images_data_reshaped = np.reshape(images_data, newshape=(len(images_data), num_channels, im_dim, im_dim)).transpose(0,2,3,1)
            images_data_reshaped = np.reshape(images_data, newshape=(len(images_data), im_dim, im_dim, num_channels))
            print('转化后格式：',images_data_reshaped.shape)
            # 将维度转换后存入数组
            dataset_array[index*images_per_file:(index+1)*images_per_file, :, :, :] = images_data_reshaped
            dataset_labels[index*images_per_file:(index+1)*images_per_file] = data_dict[b'labels']
            index += 1

    return dataset_array, dataset_labels
        


# 读取二进制文件，返回读取的信息
def unpickle_patch(file):
    patch_bin_file = open(file, 'rb')
    patch_dict = pickle.load(patch_bin_file, encoding='bytes')
    return patch_dict 


# 模型实例化函数，判断是否有预训练模型，有则优先加载预训练模型
def create_model():
    if 'pretrained_model' in gConfig:
        model = tf.keras.models.load_model(gConfig['pretrained_model'])
        return model 

    ckpt = tf.io.gfile.listdir(gConfig['working_directory'])   
    # 判断是否有模型文件
    if ckpt:
        model_file = os.path.join(gConfig['working_directory'], ckpt[-1])
        print('reading model parameters from %s' % model_file)
        model = tf.keras.models.load_model(model_file)
        return model
    else:
        model = cnnModel(rate = gConfig['rate'])
        model = model.createModel()
        return model 


# 读取训练集数据
dataset_array, dataset_labels = read_data(dataset_path=gConfig['dataset_path'],
                                        im_dim = gConfig['im_dim'], num_channels = gConfig['num_channels'], 
                                        num_files = gConfig['num_files'], images_per_file = gConfig['images_per_file'])
# 对训练数据归一化处理
dataset_array = dataset_array.astype('float32')/255
# 对标注数据进行one-hot编码 
dataset_labels = tf.keras.utils.to_categorical(dataset_labels, 10)      # 应该也会常用


# 定义训练函数
def train():
    model = create_model()
    print(model.summary())
    # 开始模型训练
    history = model.fit(dataset_array, dataset_labels, batch_size = gConfig['batch_size'], verbose = 1, epochs = 50, validation_split = 0.1)
    # 将完成训练的模型保存起来
    filename = 'cnn_model.h5'
    checkpoint_path = os.path.join(gConfig['working_directory'], filename)
    model.save(checkpoint_path)
    sys.stdout.flush()

def predict(data):
    # 获取模型文件路径
    checkpoint_path = os.path.join(gConfig['working_directory'], 'cnn_model.h5')
    model = tf.keras.models.load_model(checkpoint_path)
    prediction = model.predict(data)
    index = tf.math.argmax(prediction[0]).numpy()
    return index


if __name__ == '__main__':
    gConfig = getConfig.get_config(config_file = 'config.ini')
    if gConfig['mode'] == 'train':
        train()
    elif gConfig['mode'] == 'server':
        print('please use: python3 app.py')
