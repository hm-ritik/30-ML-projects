print("Handwritten Digit Recogination")
print("-------------------------------")

"We are making a ml model for recognising Handwritten Digit using Keras/Tensorflow"

"This is a deep learning Project which include Neural Network"
"Work Flow"
"Load dataset -> Normalize -> model building -> Compile the model -> Fit model -> Predictions "

" we are going to build our first deep learning project"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , models
from tensorflow.keras.models import Sequential

"fetching dataset for the building model"
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#normalization to make std  between 0-1
x_train=x_train/255.0
x_test=x_test/255.0

x_train=x_train.reshape(-1,28,28,1)
x_test=x_test.reshape(-1,28,28,1)

print(x_train.shape)
print(x_test.shape)
x=int(input("enter the Index number"))
import matplotlib.pyplot as plt 
plt.imshow(x_train[x] , camp='grey)
plt.title(" Label" + x_train[x])
plt.show()           

"building model by using CNN"
"conv2d to make model able to detect edge , shape of the  image"
"maxpool2d to make the image small (selecting neccessary features)"
"flatten to make image 1d "
"Dense for neural nwtwork connection (relu=Non_linear datapoints)"
"softmax for number of output"


model=models.Sequential([
      layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
      layers.MaxPool2D((2,2)),

      layers.Conv2D(64 , (3,3) , activation='relu'),
      layers.MaxPooling2D((2,2)),

      layers.Flatten(),
      layers.Dense(128 , activation='relu'),
      layers.Dense(10 , activation='softmax')

])

"Compile the model"

"Adam (self optimizer)"
"loss= for calculating loss"
"for calculating accuracy "

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

"Train the model"
model.fit(x_train,y_train, epochs=8 ,validation_data=(x_test, y_test) )
test_loss, test_acc=model.evaluate(x_test,y_test, verbose=2)

print(test_acc)
