import tensorflow as tf
import numpy as np
import getConfig 

gConfig = {}
gConfig = getConfig.get_config(config_file = 'config.ini')

# 定义cnnModel方法类
class cnnModel(object):
    def __init__(self,rate):
        self.rate=rate
    def createModel(self):
       
        model = tf.keras.Sequential()
        
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                    input_shape=(32,32,3),name="conv1"))
        model.add(tf.keras.layers.Conv2D(64, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                    name="conv2"))
        
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool1"))

        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d1"))
     
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv3"))
        model.add(tf.keras.layers.Conv2D(128, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                    name="conv4"))
        
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool2"))
  
        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d2"))
       
        model.add(tf.keras.layers.BatchNormalization())
        
        model.add(tf.keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv5"))
        model.add(tf.keras.layers.Conv2D(256, 3, kernel_initializer='he_normal', strides=1, activation='relu', padding='same',
                                   name="conv6"))
        
        model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2, padding='valid', name="pool3"))

        model.add(tf.keras.layers.Dropout(rate=self.rate, name="d3"))
        
        model.add(tf.keras.layers.BatchNormalization())
       
        model.add(tf.keras.layers.Flatten(name="flatten"))
       
        model.add(tf.keras.layers.Dropout(self.rate))
       
        model.add(tf.keras.layers.Dense(128, activation='relu',kernel_initializer='he_normal'))
       
        model.add(tf.keras.layers.Dropout(self.rate))
        
        model.add(tf.keras.layers.Dense(10, activation='softmax',kernel_initializer='he_normal'))
        
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
       
        return model

    # define dropout rate
    # def __init__(self, rate, learning_rate):
    #     self.rate = rate

    # # define a neral network model, use tf.keras.Sequential
    # def createModel(self):
    #     model = tf.keras.Sequential()
    #     # use 2d conv nn
    #     model.add(tf.keras.layers.Conv2D(64,(3,3),  strides = 1, padding = 'same',
    #                                         activation = 'relu', input_shape=[32,32,3], name = 'conv1'))
    #     #model.add(tf.keras.layers.Conv2D(64,(3,3), kernel_initializer = 'he_normal', strides = 1, padding = 'same',
    #     #                                    activation = 'relu', name = 'conv2'))
    #     # 2-d max pooling
    #     model.add(tf.keras.layers.MaxPool2D((2,2), strides = 2, padding = 'valid', name = 'pool1'))
    #     model.add(tf.keras.layers.Dropout(rate = 0.2, name = 'd1'))
    #     # add Batch Normalization
    #     model.add(tf.keras.layers.BatchNormalization())
    #     # secd conv nn
    #     model.add(tf.keras.layers.Conv2D(128,(3,3), strides = 1, padding = 'same',
    #                                         activation = 'relu', name = 'conv3'))
    #     #model.add(tf.keras.layers.Conv2D(128,(3,3), kernel_initializer = 'he_normal', strides = 1, padding = 'same',
    #     #                                    activation = 'relu', name = 'conv4'))
    #     model.add(tf.keras.layers.MaxPool2D((2,2), strides = 1, padding = 'valid', name = 'pool2'))
    #     model.add(tf.keras.layers.Dropout(rate = self.rate, name = 'd2'))

    #     model.add(tf.keras.layers.BatchNormalization())

    #     # #thrid conv
    #     # model.add(tf.keras.layers.Conv2D(256,(3,3), kernel_initializer = 'he_normal', strides = 1, padding = 'same',
    #     #                                     activation = 'relu', name = 'conv5'))
    #     # #model.add(tf.keras.layers.Conv2D(256,(3,3), kernel_initializer = 'he_normal', strides = 1, padding = 'same',
    #     # #                                    activation = 'relu', name = 'conv6'))
    #     # model.add(tf.keras.layers.MaxPool2D((2,2), strides = 1, padding = 'same', name = 'pool3'))
    #     # model.add(tf.keras.layers.Dropout(rate = self.rate, name = 'd3'))
    #     # model.add(tf.keras.layers.BatchNormalization())

    #     # Flatten operation
    #     model.add(tf.keras.layers.Flatten(name = 'flatten'))
    #     model.add(tf.keras.layers.Dense(512, activation='relu',kernel_initializer='he_normal'))
    #     model.add(tf.keras.layers.Dense(512, activation='relu',kernel_initializer='he_normal'))
    #     model.add(tf.keras.layers.Dense(10, activation = 'softmax'))
    #     # compile
    #     model.compile(loss = 'categorical_crossentropy', optimizer = tf.keras.optimizers.Adam(), metrics = ['accuracy'])

    #     return model


if __name__ == '__main__':
    cnn1 = cnnModel(0.5, 0.01)