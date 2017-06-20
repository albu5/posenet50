from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.layers import Dense, Input, Reshape
from keras.models import Model, load_model
from keras.optimizers import adam
import os
from numpy import genfromtxt
from matplotlib import pyplot as plt
from utils import get_image_names_and_labels, get_parse_batch, get_parse_batch_test
from random import randint
from keras import backend as Keras
import numpy as np
import keras


'''
========================================================================================================================
CONSTANTS
========================================================================================================================
'''
batch_size = 32
num_iter = 10000000
decay_step = 10000000
save_step = 1000
disp_step = 10
eval_step = 10
model_path = './models-posenet50/total_loss_all_aug'


'''
========================================================================================================================
CUSTOM LOSSES HERE
========================================================================================================================
'''


def good_loss(y_true, y_pred):
    return Keras.mean(-y_true * Keras.log(y_pred + Keras.epsilon()), axis=-1)


def bad_loss(y_true, y_pred):
    cost_matrix_np = np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
                               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
                               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                               [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                               [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                               ], dtype=np.float32)
    cost_matrix = Keras.constant(value=cost_matrix_np, dtype=Keras.floatx())
    bad_pred = Keras.dot(y_true, cost_matrix)
    return Keras.mean(-bad_pred * Keras.log(1 - y_pred + Keras.epsilon()), axis=-1)


def total_loss(y_true, y_pred):
    return bad_loss(y_true, y_pred) + good_loss(y_true, y_pred)

keras.losses.good_loss = good_loss
keras.losses.total_loss = total_loss

'''
========================================================================================================================
MODEL DEFINITION
========================================================================================================================
'''
# resnet = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
#
# # for layer in resnet.layers:
# #     layer.trainable = False
#
# input_layer = Input(shape=(224, 224, 3))
# resnet_features = resnet(input_layer)
# # resnet_flat = Keras.squeeze(Keras.squeeze(resnet_features, axis=1), axis=1)
# resnet_features = Reshape(target_shape=(2048, ))(resnet_features)
# resnet_dense = Dense(1024, activation='relu')(resnet_features)
# resnet_prob = Dense(8, activation='softmax')(resnet_dense)
# pose_resnet = Model(input=input_layer, output=resnet_prob)
#
optimizer = adam(lr=0.00001)
# pose_resnet.compile(optimizer=optimizer, loss=total_loss, metrics=['accuracy'])
# print(pose_resnet.summary())

pose_resnet = load_model(model_path)
'''
========================================================================================================================
TRAINING
========================================================================================================================
'''
data_path = './../tf-vgg/PARSE224/test'
img_names, labels_vec = get_image_names_and_labels(data_path)

x, y = get_parse_batch_test(img_names, labels_vec, 224, 224, 0, batch_size)
print(x.shape, y.shape)
# plt.imshow((x[0, :, :, :] - x.min())/(x.max() - x.min()))
# plt.pause(2)


acc_arr = []
i = 0

while i < 6599-batch_size:
    X, Y = get_parse_batch_test(img_names, labels_vec, 224, 224, start_i=i, end_i=i+batch_size)
    y = pose_resnet.predict(X)
    if i == 0:
        y_true = Y
        y_pred = y
    else:
        y_true = np.vstack((y_true, Y))
        y_pred = np.vstack((y_pred, y))
    i += batch_size
    print(i*100/6599, '% completed', Y.shape, y.shape, y_true.shape, y_pred.shape)

np.savetxt('./../tf-vgg/PARSE224/res/test_total_loss_all_aug_true.txt', y_true, delimiter=',')
np.savetxt('./../tf-vgg/PARSE224/res/test_total_loss_all_aug_pred.txt', y_pred, delimiter=',')
