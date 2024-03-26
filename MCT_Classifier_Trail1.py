#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:52:55 2024

@author: user
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import os
import numpy as np
path_data = 'Sign_Real_Data_2'
#%%
"""Load Motion Capture Data from .mat files consisting of reference 3D 
joints of signs"""
"""Extract Labels from the file names of .mat Motion capture data"""
import scipy.io
from scipy.interpolate import interp1d
Motion_Capture_DataPath = os.path.join(path_data,'MCT_Signs_100')
mat_file_list = os.listdir(Motion_Capture_DataPath)
MCT_Labels = []
new_size = 60
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
        interpolation_function = interp1d(np.arange(Final_MCT_Joints.shape[0]), Final_MCT_Joints, kind='linear', axis=0)
        Final_MCT_Joints1 = interpolation_function(np.linspace(0, Final_MCT_Joints.shape[0] - 1, new_size))
        MCT_joints.append(Final_MCT_Joints1)
        MCT_data_dict['MCT_joints'].append(Final_MCT_Joints)
        MCT_data_dict['Labels'].append(label_map_mct[extract_mat_lables_1])
"""Check if the data in both MCT and MP are same"""
print(MCT_data_dict['MCT_Labels'])
#%%
"""Unpack x,y,z coordinates"""
MCT_Joints = np.array(MCT_joints)
print(MCT_Joints.shape)
grouped_indices = [np.arange(start,171, step=3) for start in range(3)]
Reshaped_MCT_Joints = np.empty((8,60,len(grouped_indices),57))
for i, indices in enumerate(grouped_indices):
    Reshaped_MCT_Joints[:,:,i,:] = MCT_Joints[:,:,indices]
Reshaped_MCT_Joints_1 = np.swapaxes(Reshaped_MCT_Joints, 2, 3)
print(Reshaped_MCT_Joints_1.shape)
labels = np.array(MCT_data_dict['Labels'])
MCT_Joints_Train = np.concatenate([Reshaped_MCT_Joints_1[:,:,:,i].reshape(8,60,-1) for i in range(3)], axis = -1)
print(MCT_Joints_Train.shape)
#%%
"""Train a LSTM based classifier"""
import tensorflow as tf
from tensorflow import keras
from keras import callbacks, models
n_classes = 8
X_train = MCT_Joints
y_train1 = MCT_data_dict['Labels']
y_train = tf.keras.utils.to_categorical(y_train1, n_classes)
model = keras.Sequential([
    keras.layers.LSTM(128, return_sequences = True, input_shape =(60,171)),
    keras.layers.LSTM(64, return_sequences = True),
    keras.layers.LSTM(32),
    keras.layers.Dense(n_classes,activation = 'softmax')
    ])
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#%%
model.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
checkpoint_filepath = 'MCT_Data_Classifier_Trained_Model_1.h5'
model_checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath,
                                             monitor='loss',
                                             save_best_only=True,
                                             mode = 'min',
                                             verbose = 1)
history = model.fit(X_train, y_train, 
          epochs = 200, batch_size = 16,
          callbacks = [model_checkpoint])
#%%
"""Evaluate the model with the saved BEST Model"""
best_model = models.load_model(checkpoint_filepath)
test_loss, test_acc = best_model.evaluate(X_train, y_train)
print(f'Test accuracy: {test_acc}')
#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_train)
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_train1, y_pred_labels)
print('\nConfusion Matrix:')
print(conf_matrix)

