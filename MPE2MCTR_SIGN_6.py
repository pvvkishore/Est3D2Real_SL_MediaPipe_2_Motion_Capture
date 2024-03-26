#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 15:52:43 2024

@author: user
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 15:30:31 2024

@author: user
"""

"""Media_Pipe Estimates Reconstruction with Motion Capture References"""
import os
import numpy as np
"""Read media pipe recorded 3D skeletal data from folder"""
path_data = 'Sign_Real_Data_2'
Data_path_for_mediapipe = os.path.join(path_data, 'MediaPipe_Estimated_Data')

Sign_Classes = [d for d in os.listdir(Data_path_for_mediapipe) if os.path.isdir(os.path.join(Data_path_for_mediapipe,d))]
actions = np.array(Sign_Classes)
#print(actions)

# Load Media Pipe Data from .npy files
label_map = {lable:num for num, lable in enumerate(actions)}
mp_data_dict = {'sequences':[],'labels':[], 'actions':Sign_Classes}
#sequences, labels = [],[]
number_of_sequences = 8
sequence_length = 60
for action in actions:
    for sequence in np.array(os.listdir(os.path.join(Data_path_for_mediapipe, action))).astype(int):
        window = []
        for frame_number in range(sequence_length):
            res_mp = np.load(os.path.join(Data_path_for_mediapipe,action,
                             str(sequence), "{}.npy".format(frame_number)))
            res_mp1 = res_mp.reshape((171), order = 'F')
            window.append(res_mp1)
        mp_data_dict['sequences'].append(window)
        mp_data_dict['labels'].append(label_map[action])
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
group_indices = [np.arange(i,171,3) for i in range(3)]
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
        reshaped_tensor = Final_MCT_Joints1[:,:3*57].reshape((60, 57, 3), order='C')
        Final_MCT_Joints2 = reshaped_tensor.reshape((60,171), order = 'F')
        MCT_data_dict['MCT_joints'].append(Final_MCT_Joints2)
        MCT_data_dict['Labels'].append(label_map_mct[extract_mat_lables_1])
"""Check if the data in both MCT and MP are same"""
print(MCT_data_dict['MCT_Labels'] == mp_data_dict['actions'])
# If the result of the above statement is 'FALSE' perfrom the next step
#%%
"""Pair simialr labels and their associated data as input and targets
    for training the embedding layer"""
threshold = 3
label_pairs = {}

for label1 in mp_data_dict['actions']:
    closest_lable, max_match = None, 0
    for label2 in MCT_data_dict['MCT_Labels']:
        match_count = sum(1 for c1,c2 in zip(label1, label2) if c1 == c2)
        if match_count > max_match:
            closest_lable, max_match = label2, match_count
    print(label1,closest_lable)        
    if max_match >= threshold:
        data1 = mp_data_dict[list(mp_data_dict.keys())[0]][mp_data_dict['actions'].index(label1)]
        data2 = MCT_data_dict[list(MCT_data_dict.keys())[0]][MCT_data_dict['MCT_Labels'].index(closest_lable)]
        
        label_pairs[(label1, closest_lable)] = (np.array(data1), data2)
        #%%
"""Split each 3D point into 1D x,y,z axis and construct 3 training pairs
also scale the MCT data """
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

training_data_pairs = {}
for (label1,closest_lable),(data1, data2) in label_pairs.items():
    training_data_pairs[(label1,closest_lable)] = (data1, ((data2-np.min(data2))/(np.max(data2)-np.min(data2))))
#%%
import tensorflow as tf
from tensorflow.keras import layers, Model
class MPE2MCT_MAPPER(Model):
    def __init__(self):
        super(MPE2MCT_MAPPER,self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(256, activation='relu')
        self.dense3 = layers.Dense(60*171, activation='linear')
        
    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)
model = MPE2MCT_MAPPER()
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.000001)
model.compile(optimizer = 'adam',
              loss = 'mean_squared_error',
              metrics = ['mae'])
# Check if the model is compiled properly
if model.loss is None:
    raise ValueError("Model is not compiled properly. Please check the compilation step.")
#model.summary()
#%%
for (_,_), (MP_input, MCT_Output) in training_data_pairs.items():
    #print(MP_input.shape)
    MP_input = np.expand_dims(MP_input, axis = 0)
    #print(MP_input.shape)
    MCT_Output = np.expand_dims(MCT_Output, axis = 0)
    MCT_Output = np.reshape(MCT_Output, (1,10260))
    #print(MCT_Output.shape)
    model.fit(MP_input, MCT_Output, epochs = 200, batch_size = 1)
#%% Get predictions for each data pair
MCT_predictions = {}
MP_2_MCT_converted = []
for (key,_),(input_array, _) in training_data_pairs.items():
    input_array = np.expand_dims(input_array, axis=0)
    print(key)
    predictions = model.predict(input_array)
    predictions = np.reshape(predictions, (60,171))
    MCT_predictions[key] = predictions
    #MCT_predictions[key] = (predictions - np.min(predictions))/(np.max(predictions)-np.min(predictions))
#%%
"""Classify the predictions using the best model"""
Predicted_MCT = np.stack([value for value in MCT_predictions.values()])
#%%
Load_MCT_Classifier = tf.keras.models.load_model("MCT_Data_Classifier_Trained_Model_1.h5")
predicted_MP_MCT = Load_MCT_Classifier.predict(Predicted_MCT)
#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#y_pred = model.predict(X_train)
y_train1 = MCT_data_dict['Labels']
y_pred_labels = np.argmax(predicted_MP_MCT, axis=1)
conf_matrix = confusion_matrix(y_train1, y_pred_labels)
print('\nConfusion Matrix:')
print(conf_matrix)
#%%
"""Train a LSTM Classifier on the Predicted_MP_MCT Data"""
#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
"""Train a LSTM based classifier"""
import tensorflow as tf
from tensorflow import keras
from keras import callbacks, models
n_classes = 8
X_train = Predicted_MCT
y_train1 = MCT_data_dict['Labels']
y_train = tf.keras.utils.to_categorical(y_train1, n_classes)
model_1 = keras.Sequential([
    keras.layers.LSTM(128, return_sequences = True, input_shape =(60,171)),
    keras.layers.LSTM(64, return_sequences = True),
    keras.layers.LSTM(32),
    keras.layers.Dense(n_classes,activation = 'softmax')
    ])
optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
#%%
model_1.compile(loss = 'categorical_crossentropy',optimizer = 'adam',
              metrics = ['accuracy'])
checkpoint_filepath = 'MP_2_MCT_Data_Classifier_Trained_Model.h5'
model_checkpoint = callbacks.ModelCheckpoint(checkpoint_filepath,
                                             monitor='loss',
                                             save_best_only=True,
                                             mode = 'min',
                                             verbose = 1)
history = model_1.fit(X_train, y_train, 
          epochs = 200, batch_size = 32,
          callbacks = [model_checkpoint])
#%%
"""Evaluate the model with the saved BEST Model"""
best_model = models.load_model(checkpoint_filepath)
test_loss, test_acc = best_model.evaluate(X_train, y_train)
print(f'Test accuracy: {test_acc}')
#%%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model_1.predict(X_train)
y_pred_labels = np.argmax(y_pred, axis=1)
conf_matrix = confusion_matrix(y_train1, y_pred_labels)
print('\nConfusion Matrix:')
print(conf_matrix)
    
    
    
    
    
    
    
    
    
    
    