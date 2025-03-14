import numpy as np
import matplotlib.pyplot as plt
import csv as cv

from keras.src.layers.activations import activation
from tensorflow.keras import datasets , layers, models
from tensorflow.python.keras.saving.saved_model.load import metrics

(training_images,training_labels) , (testing_images, testing_labels)=datasets.cifar10.load_data()
training_images,testing_images= training_images / 255, testing_images / 255

class_names = ['Plane','Car','Bird','Cat','Deer','Dog','Frog','Horse','Ship','Trunk']

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images=testing_images[:20000]
testing_labels=testing_labels[:20000]

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(training_images,training_labels,epochs=10)