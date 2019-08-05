#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 20:56:49 2019

@author: mridhoula
"""

import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import  Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D


import pandas as pd
from PIL import Image
import os
import numpy as np
dimen = 64
input_shape  = (dimen,dimen,1)
inp_img = Input(shape = (dimen,dimen,1), name = 'ImageInput')
model = inp_img


model = Conv2D(32,kernel_size=(3, 3),activation='relu',input_shape=input_shape,padding='valid')(model)

model = MaxPooling2D((2,2), padding='valid')(model)
model = Conv2D(64, (3, 3), activation='relu',padding='valid')(model)

model = MaxPooling2D((2,2),padding='valid')(model)

model = Conv2D(128, (3, 3), activation='relu',padding='valid')(model)
model = MaxPooling2D((2,2),padding='valid')(model)


model = Conv2D(256, (1, 1), activation='relu',padding='valid')(model)
model = MaxPooling2D((2,2),padding='valid')(model)

model = Conv2D(64, (1, 1), activation='relu',padding='valid')(model)

model = Flatten()(model)



e = Model(inputs=[inp_img], outputs=[model],name = 'Feat_Model')
e.summary()


# In[27]:

left_img = Input(shape = (dimen,dimen,1), name = 'left_img')
right_img = Input(shape = (dimen,dimen,1), name = 'right_img')

# In[28]:

left_e = e(left_img)
right_e = e(right_img)


# In[35]:

from keras.layers import concatenate
from keras.layers.normalization import BatchNormalization




# In[36]:

feats_m = concatenate([left_e, right_e], name = 'concat_feats')
feats_m = Dense(1024, activation = 'linear')(feats_m)
feats_m = BatchNormalization()(feats_m)
feats_m = Activation('relu')(feats_m)
feats_m = Dense(4, activation = 'linear')(feats_m)
feats_m = BatchNormalization()(feats_m)
feats_m = Activation('relu')(feats_m)
feats_m = Dense(1, activation = 'sigmoid')(feats_m)
model1= Model(inputs = [left_img, right_img], outputs = [feats_m], name = 'Model')
model1.summary()




train_x = []
train_y  = []
train_target=[]
df = pd.read_csv('dataset_seen_training_siamese.csv')

a=[]
b=[]
imgs = []
train_target=[]
valid_images = [".png"]

curPath= 'TrainingSetseen' 
imgs =  os.listdir(curPath)
'''Training set'''

'''Target'''

for row in df['label']:
    train_target.append(row)

train_target1=np.array(train_target)
train_target2=train_target1.reshape(len(train_target1),1)
print(train_target2.shape)

'''left'''
for row in df['left']:
    a.append(row)

for i in a:
    
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2 = img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               train_x.append(imgdata)
               break
            
image_vector_size=64*64*1 
num_classes=2   

train_x1 = np.array(train_x)
train_x2=train_x1.reshape(len(train_x1),64,64,1)
print(train_x2.shape)
''' right'''
for row in df['right']:
    b.append(row)
for i in b:
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2= img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               train_y.append(imgdata)
               break
            
            
train_y1 = np.array(train_y)
train_y2=train_y1.reshape(len(train_y1),64,64,1)
print(train_y2.shape)           
            
            
            
            
            
'''Validation set'''    
df1 = pd.read_csv('dataset_seen_validation_siamese.csv')
curPath= 'ValidationSetseen' 
imgs =  os.listdir(curPath)   
a1=[]
b1=[]  
test_x=[]
test_y=[] 
test_target=[]
'''Target'''

for row in df1['label']:
    test_target.append(row)

test_target1=np.array(test_target)
test_target2=test_target1.reshape(len(test_target1),1)
print(test_target2.shape)

'''left'''
for row in df1['left']:
    a1.append(row)

for i in a1:
    
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2 = img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               test_x.append(imgdata)
               break
            
image_vector_size=64*64*1 
num_classes=2   

test_x1 = np.array(test_x)
test_x2=test_x1.reshape(len(test_x1),64,64,1)
print(test_x2.shape)
''' right'''
for row in df1['right']:
    b1.append(row)
for i in b1:
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2= img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               test_y.append(imgdata)
               break

        
image_vector_size=64*64*1   
test_y1 = np.array(test_y)
test_y2=test_y1.reshape(len(test_y1),64,64,1)
print(test_y2.shape)
image_size=4096
validation_data_split = 0.2
num_epochs = 10
model_batch_size = 128
tb_batch_size = 32
model1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model1.fit([train_x2,train_y2], train_target2
                   
                    , epochs=num_epochs
                    )


loss1,accuracy1 = model1.evaluate([train_x2, train_y2],train_target2, verbose=False)
print(" Training Accuracy for seen Dataset")
print(accuracy1)
print("Training Loss for seen dataset")
print(loss1)
loss,accuracy = model1.evaluate([test_x2, test_y2],test_target2, verbose=False)
print(" Validation Accuracy for seen Dataset")
print(accuracy)
print("Validation Loss for seen dataset")
print(loss)



'-------------------------------------------------------------------------------'
'''Unseen dataset'''


train_x = []
train_y  = []
train_target=[]
df = pd.read_csv('dataset_unseen_training_siamese.csv')

a=[]
b=[]
imgs = []
train_target=[]
valid_images = [".png"]

curPath= 'TrainingSetunseen' 
imgs =  os.listdir(curPath)
'''Training set'''

'''Target'''

for row in df['label']:
    train_target.append(row)

train_target1=np.asarray(train_target)
train_target2=train_target1.reshape(len(train_target1),1)
print(train_target2.shape)

'''left'''
for row in df['left']:
    a.append(row)

for i in a:
    
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2 = img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               train_x.append(imgdata)
               break
            
image_vector_size=64*64*1 
num_classes=2   

train_x1 = np.array(train_x)
train_x2=train_x1.reshape(len(train_x1),64,64,1)
print(train_x2.shape)
''' right'''
for row in df['right']:
    b.append(row)
for i in b:
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2= img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               train_y.append(imgdata)
               break
            
            
train_y1 = np.array(train_y)
train_y2=train_y1.reshape(len(train_y1),64,64,1)
print(train_y2.shape)           
            
            
            
            
            
'''Validation set'''    
df1 = pd.read_csv('dataset_unseen_validation_siamese.csv')
curPath= 'ValidationSetunseen' 
imgs =  os.listdir(curPath)   
a1=[]
b1=[]  
test_x=[]
test_y=[] 
test_target=[]
'''Target'''

for row in df1['label']:
    test_target.append(row)

test_target1=np.array(test_target)
test_target2=test_target1.reshape(len(test_target1),1)
print(test_target2.shape)

'''left'''
for row in df1['left']:
    a1.append(row)

for i in a1:
    
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2 = img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               test_x.append(imgdata)
               break
            
image_vector_size=64*64*1 
num_classes=2   

test_x1 = np.array(test_x)
test_x2=test_x1.reshape(len(test_x1),64,64,1)
print(test_x2.shape)
''' right'''
for row in df1['right']:
    b1.append(row)
for i in b1:
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2= img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               test_y.append(imgdata)
               break

        
image_vector_size=64*64*1   
test_y1 = np.array(test_y)
test_y2=test_y1.reshape(len(test_y1),64,64,1)
print(test_y2.shape)
image_size=4096
validation_data_split = 0.2
num_epochs = 10
model_batch_size = 128
tb_batch_size = 32
model1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model1.fit([train_x2,train_y2], train_target2
                   
                    , epochs=num_epochs
                    )


loss1,accuracy1 = model1.evaluate([train_x2, train_y2],train_target2, verbose=False)
print(" Training Accuracy for unseen Dataset")
print(accuracy1)
print("Training Loss for unseen dataset")
print(loss1)
loss,accuracy = model1.evaluate([test_x2, test_y2],test_target2, verbose=False)
print(" Validation Accuracy for unseen Dataset")
print(accuracy)
print("Validation Loss for unseen dataset")
print(loss)


'-------------------------------------------------------------------------------'
'''shuffled dataset'''


train_x = []
train_y  = []
train_target=[]
df = pd.read_csv('dataset_shuffled_training_siamese.csv')

a=[]
b=[]
imgs = []
train_target=[]
valid_images = [".png"]

curPath= 'TrainingSetshuffled' 
imgs =  os.listdir(curPath)
'''Training set'''

'''Target'''

for row in df['label']:
    train_target.append(row)

train_target1=np.asarray(train_target)
train_target2=train_target1.reshape(len(train_target1),1)
print(train_target2.shape)

'''left'''
for row in df['left']:
    a.append(row)

for i in a:
    
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2 = img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               train_x.append(imgdata)
               break
            
image_vector_size=64*64*1 
num_classes=2   

train_x1 = np.array(train_x)
train_x2=train_x1.reshape(len(train_x1),64,64,1)
print(train_x2.shape)
''' right'''
for row in df['right']:
    b.append(row)
for i in b:
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2= img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               train_y.append(imgdata)
               break
            
            
train_y1 = np.array(train_y)
train_y2=train_y1.reshape(len(train_y1),64,64,1)
print(train_y2.shape)           
            
            
            
            
            
'''Validation set'''    
df1 = pd.read_csv('dataset_shuffled_validation_siamese.csv')
curPath= 'ValidationSetshuffled' 
imgs =  os.listdir(curPath)   
a1=[]
b1=[]  
test_x=[]
test_y=[] 
test_target=[]
'''Target'''

for row in df1['label']:
    test_target.append(row)

test_target1=np.array(test_target)
test_target2=test_target1.reshape(len(test_target1),1)
print(test_target2.shape)


'''left'''
for row in df1['left']:
    a1.append(row)

for i in a1:
    
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2 = img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               test_x.append(imgdata)
               break
            
image_vector_size=64*64*1 
num_classes=2   

test_x1 = np.array(test_x)
test_x2=test_x1.reshape(len(test_x1),64,64,1)
print(test_x2.shape)
''' right'''
for row in df1['right']:
    b1.append(row)
for i in b1:
    
    for img in imgs:
        
        
    
        
        
        if row==img:
           
       
           curImg = curPath + '/' + img
           

           if curImg[-3:] == 'png':
               
               img = Image.open(curImg,'r')
               img1=img.convert('L')
               img2= img1.resize((64,64))
               savedImg = img2
               imgdata = (255-np.array(img2.getdata()))/255
               test_y.append(imgdata)
               break

        
image_vector_size=64*64*1   
test_y1 = np.array(test_y)
test_y2=test_y1.reshape(len(test_y1),64,64,1)
print(test_y2.shape)
image_size=4096
validation_data_split = 0.2
num_epochs = 10
model_batch_size = 128
tb_batch_size = 32
model1.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model1.fit([train_x2,train_y2], train_target2
                   
                    , epochs=num_epochs
                    )


loss1,accuracy1 = model1.evaluate([train_x2, train_y2],train_target2, verbose=False)
print(" Training Accuracy for shuffled Dataset")
print(accuracy1)
print("Training Loss for shuffled dataset")
print(loss1)
loss,accuracy = model1.evaluate([test_x2, test_y2],test_target2, verbose=False)
print(" Validation Accuracy for shuffled Dataset")
print(accuracy)
print("Validation Loss for shuffled dataset")
print(loss)