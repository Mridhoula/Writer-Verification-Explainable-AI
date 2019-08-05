#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 22:11:58 2019

@author: mridhoula
"""
import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import  Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, BatchNormalization, Activation, Reshape, UpSampling2D
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import pandas as pd
from PIL import Image
import os
import numpy as np

input_img = Input(shape=(64, 64, 1))  # adapt this if using `channels_first` image data format
# x = CoordinateChannel2D()(input_img)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
# x = CoordinateChannel2D()(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# x = CoordinateChannel2D()(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)
encoded = Flatten()(x)
encoded = Dense(8*8*8, activation='relu', name='latent')(encoded)
# model = Model(input_img,encoded)
# print(model.summary())
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
r = Reshape(target_shape=(8,8,8))(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(r)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same',name='output')(x)

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
autoencoder.summary()


encoder = Model(inputs=autoencoder.inputs,outputs=autoencoder.get_layer('latent').output)
encoder.summary()

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

history = autoencoder.fit(train_x2, train_x2
                   
                    , epochs=num_epochs
                    )
history = autoencoder.fit(train_y2, train_y2
                   
                    , epochs=num_epochs
                    )
py1 = autoencoder.predict(test_x2)
py2 = autoencoder.predict(test_y2)
print(py1)
print(py1.shape,py2.shape)
list=[]
for i in range(906):
    
    
    dot_product = np.dot(py1[i], py2[i])
    
    norm_a = np.linalg.norm(py1[i])
    
    norm_b = np.linalg.norm(py2[i])
    
    cs=dot_product / (norm_a * norm_b)
    
    list.append(cs)


print(min(list))
en_val=[]
for i in range(906):
    
    if list[i] <= 0.5:
       encode_val=0
       en_val.append(encode_val)
    else:
        encode_val=1
        en_val.append(encode_val)
     
print(encode_val)


from keras.layers import Dense
from keras.models import Sequential
'''Pen Pressure'''
num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Letter Spacing'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''size'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Dimension'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


      
'''Lower case'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Continuos'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Slantness'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''tilt'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Entry stoke'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Staff  of a'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Formation  of n'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Staff of d'''

num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Exit Stroke_D'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''word formation'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Constancy'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)



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
history = autoencoder.fit(train_x2, train_x2
                   
                    , epochs=num_epochs
                    )
history = autoencoder.fit(train_y2, train_y2
                   
                    , epochs=num_epochs
                    )
py1 = autoencoder.predict(test_x2)
py2 = autoencoder.predict(test_y2)
print(py1)
print(py1.shape,py2.shape)
list=[]
for i in range(906):
    
    
    dot_product = np.dot(py1[i], py2[i])
    
    norm_a = np.linalg.norm(py1[i])
    
    norm_b = np.linalg.norm(py2[i])
    
    cs=dot_product / (norm_a * norm_b)
    
    list.append(cs)


print(min(list))
en_val=[]
for i in range(906):
    
    if list[i] <= 0.5:
       encode_val=0
       en_val.append(encode_val)
    else:
        encode_val=1
        en_val.append(encode_val)
     
print(encode_val)


from keras.layers import Dense
from keras.models import Sequential
'''Pen Pressure'''
num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Letter Spacing'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''size'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Dimension'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


      
'''Lower case'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Continuos'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Slantness'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''tilt'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Entry stoke'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Staff  of a'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Formation  of n'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Staff of d'''

num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Exit Stroke_D'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''word formation'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Constancy'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


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
history = autoencoder.fit(train_x2, train_x2
                   
                    , epochs=num_epochs
                    )
history = autoencoder.fit(train_y2, train_y2
                   
                    , epochs=num_epochs
                    )
py1 = autoencoder.predict(test_x2)
py2 = autoencoder.predict(test_y2)
print(py1)
print(py1.shape,py2.shape)
list=[]
for i in range(906):
    
    
    dot_product = np.dot(py1[i], py2[i])
    
    norm_a = np.linalg.norm(py1[i])
    
    norm_b = np.linalg.norm(py2[i])
    
    cs=dot_product / (norm_a * norm_b)
    
    list.append(cs)


print(min(list))
en_val=[]
for i in range(906):
    
    if list[i] <= 0.5:
       encode_val=0
       en_val.append(encode_val)
    else:
        encode_val=1
        en_val.append(encode_val)
     
print(encode_val)


from keras.layers import Dense
from keras.models import Sequential
'''Pen Pressure'''
num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Letter Spacing'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''size'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Dimension'''
num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


      
'''Lower case'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Continuos'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Slantness'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''tilt'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Entry stoke'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Staff  of a'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Formation  of n'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''Staff of d'''

num_classes=3
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Exit Stroke_D'''

num_classes=4
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)

'''word formation'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)


'''Constancy'''

num_classes=2
input_size=906
model = Sequential()
model.add(Dense(units=32, activation='sigmoid', input_shape=(input_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit([py1,py2],train_target2, epochs=10)
loss,accuracy=model.predict([test_x1,test_y2],test_target2)