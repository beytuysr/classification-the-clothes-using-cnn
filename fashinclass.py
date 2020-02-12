# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:12:12 2020
fashion class classification
@author: Beytu
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#1.import the data

fashion_train_df=pd.read_csv('fashion-mnist_train.csv',sep=',')
fashion_test_df=pd.read_csv('fashion-mnist_test.csv',sep=',')

#2.visualization of the data test
fth=fashion_train_df.head()
ftt=fashion_train_df.tail()
fts=fashion_train_df.shape

# Create training and testing arrays
training=np.array(fashion_train_df,dtype='float32')
testing=np.array(fashion_test_df,dtype='float32')

# Let's view some images!
i=random.randint(1,60000)
plt.imshow(training[i,1:].reshape((28,28)))
plt.imshow(training[i,1:].reshape((28,28)),cmap='gray')
label=training[i,0]

# Let's view more images in a grid format
# Define the dimensions of the plot grid 
W_grid=15
L_grid=15

# fig, axes=plt.subplots(L_grid, W_grid)
#subplot return the figure object and axes object
#we can use the axes object to plot specific figures at various locations

fig,axes=plt.subplots(L_grid, W_grid, figsize = (17,17))
axes=axes.ravel() # flaten the 15 x 15 matrix into 225 array
n_training=len(training) # get the length of the training dataset

# Select a random number from 0 to n_training
for i in np.arange(0, W_grid * L_grid): # create evenly spaces variables 

    # Select a random number
    index=np.random.randint(0, n_training)
    # read and display an image with the selected index    
    axes[i].imshow( training[index,1:].reshape((28,28)) )
    axes[i].set_title(training[index,0], fontsize = 8)
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.4)

# Remember the 10 classes decoding is as follows:
# 0 => T-shirt/top
# 1 => Trouser
# 2 => Pullover
# 3 => Dress
# 4 => Coat
# 5 => Sandal
# 6 => Shirt
# 7 => Sneaker
# 8 => Bag
# 9 => Ankle boot

#3.training the model
X_train=training[:,1:]/255
y_train=training[:,0]

X_test=testing[:,1:]/255
y_test=testing[:,0]

from sklearn.model_selection import train_test_split
X_train,X_validate,y_train,y_validate=train_test_split(X_train, y_train, test_size = 0.2, random_state = 12345)

X_train=X_train.reshape(X_train.shape[0],*(28,28,1))
X_test=X_test.reshape(X_test.shape[0],*(28,28,1))
X_validate=X_validate.reshape(X_validate.shape[0],*(28,28,1))

Xts=X_train.shape

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

cnn_model=Sequential()
cnn_model.add(Conv2D(32,3,3,input_shape=(28,28,1),activation='relu'))
cnn_model.add(MaxPooling2D(pool_size=(2,2)))
cnn_model.add(Flatten())
cnn_model.add(Dense(output_dim=32,activation='relu'))
cnn_model.add(Dense(output_dim=10,activation='sigmoid'))

cnn_model.compile(loss='sparse_categorical_crossentropy',optimizer=Adam(lr=0.001),metrics=['accuracy'])
cnn_model.fit(X_train,y_train,batch_size=512,epochs=2,verbose=1,validation_data=(X_validate,y_validate))

#4.evaluating the model
evaluation=cnn_model.evaluate(X_test,y_test)
print('Test Accuracy : {:.3f}'.format(evaluation[1]))

# get the predictions for the test data
predicted_classes=cnn_model.predict_classes(X_test)

L=5
W=5
fig, axes=plt.subplots(L, W, figsize = (12,12))
axes=axes.ravel() 

for i in np.arange(0, L * W):  
    axes[i].imshow(X_test[i].reshape(28,28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')

plt.subplots_adjust(wspace=0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14,10))
sns.heatmap(cm, annot=True)
# Sum the diagonal element to get the total true correct values  

from sklearn.metrics import classification_report

num_classes=10
target_names=["Class {}".format(i) for i in range(num_classes)]

print(classification_report(y_test, predicted_classes, target_names=target_names))    

            