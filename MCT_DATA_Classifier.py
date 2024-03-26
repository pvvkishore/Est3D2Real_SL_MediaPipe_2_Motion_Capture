#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 09:35:25 2024

@author: user
"""

import os
import numpy as np
path_data = 'Sign_Real_Data_2'
#%%
"""Load Motion Capture Data from .mat files consisting of reference 3D 
joints of signs"""
"""Extract Labels from the file names of .mat Motion capture data"""
import scipy.io
Motion_Capture_DataPath = os.path.join(path_data,'MCT_Signs_100')
mat_file_list = os.listdir(Motion_Capture_DataPath)
MCT_Labels = []
for file_name in mat_file_list:
    if file_name.endswith('.mat'):
        extract_mat_lables = file_name.split('.')[0]
        extract_mat_lables_1 = extract_mat_lables.split('_')[0]
    MCT_Labels.append(extract_mat_lables_1.lower())
"""Load the data from .mat files""" 
label_map_mct = {lable:num for num, lable in enumerate(MCT_Labels)}
MCT_data_dict = {'MCT_joints':[],'Labels':[], 'MCT_Labels':MCT_Labels}
MCT_joints = []
for file_name in mat_file_list:
    if file_name.endswith('.mat'):
        file_path = os.path.join(Motion_Capture_DataPath,file_name)
        mat_data = scipy.io.loadmat(file_path)
        extract_mat_lables = file_name.split('.')[0]
        extract_mat_lables_1 = extract_mat_lables.split('_')[0]
        extract_mat_lables_1 = extract_mat_lables_1.lower()
        Final_MCT_Joints = np.tile(mat_data[extract_mat_lables][:,:-6],(1,1))
        MCT_joints.append(Final_MCT_Joints)
        MCT_data_dict['MCT_joints'].append(Final_MCT_Joints)
        MCT_data_dict['Labels'].append(label_map_mct[extract_mat_lables_1])
"""Check if the data in both MCT and MP are same"""
print(MCT_data_dict['MCT_Labels'])
#%%
"""Unpack x,y,z coordinates"""
MCT_Joints = np.array(MCT_joints)
print(MCT_Joints.shape)
grouped_indices = [np.arange(start,171, step=3) for start in range(3)]
Reshaped_MCT_Joints = np.empty((8,100,len(grouped_indices),57))
for i, indices in enumerate(grouped_indices):
    Reshaped_MCT_Joints[:,:,i,:] = MCT_Joints[:,:,indices]
Reshaped_MCT_Joints_1 = np.swapaxes(Reshaped_MCT_Joints, 2, 3)
print(Reshaped_MCT_Joints_1.shape)
labels = np.array(MCT_data_dict['Labels'])
MCT_Joints_Train = np.concatenate([Reshaped_MCT_Joints_1[:,:,:,i].reshape(8,100,-1) for i in range(3)], axis = -1)
print(MCT_Joints_Train.shape)
#%%
"""Train a LSTM based classifier"""
import tensorflow as tf
from tensorflow import keras
n_classes = 8
X_train = MCT_Joints
y_train = np.array(MCT_data_dict['Labels'])
#X_train = MCT_Joints.reshape(-1,100,171)
#y_train = tf.keras.utils.to_categorical(y_train, n_classes)
print(X_train.shape)
#%%
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences = True, input_shape =(100,171)),
    keras.layers.LSTM(64),
    keras.layers.Dense(n_classes,activation = 'softmax')
    ])
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
#%%
model.fit(X_train, y_train, epochs = 10, batch_size = 32)
#%%
# Define the LSTM model
from tensorflow.keras import models,layers
model = models.Sequential()
model.add(layers.LSTM(128, input_shape=(100, 171), return_sequences=True))
model.add(layers.LSTM(64, return_sequences=True))
model.add(layers.LSTM(32))
model.add(layers.Dense(8, activation='softmax'))  # Output layer with 8 classes
model.summary()


