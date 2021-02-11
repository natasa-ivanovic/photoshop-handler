import os
import warnings
import cv2
import keras
import matplotlib.pyplot as plt
import matplotlib.style as style
import numpy as np
import pandas as pd
from PIL import Image
from keras import models, layers, optimizers
from keras.applications import VGG16
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

gestures_map = {
    'brush': 0,
    'zoom': 1,
    'nav': 2,
    'delete': 3,
    'move': 4
}

image_size = 224

def process_image(path):
    img = Image.open(path)
    img = img.resize((image_size, image_size))
    img = np.array(img)
    return img

def process_data(gestures_data, labels_data):
    gestures_data = np.array(gestures_data, dtype = 'float32')
    gestures_data = np.stack((gestures_data,)*3, axis=-1)
    gestures_data /= 255
    labels_data = np.array(labels_data)
    labels_data = to_categorical(labels_data)
    return gestures_data, labels_data

def load_data(directory):
    gestures_data = []
    labels_data = []
    for filename in os.listdir(directory):
        path = directory + '/' + filename
        gesture_name = filename.split("_")[0]
        gestures_data.append(process_image(path))
        labels_data.append(gestures_map[gesture_name])

    gestures_data, labels_data = process_data(gestures_data, labels_data)
    return gestures_data, labels_data

if __name__ == '__main__':
    gestures_data, labels_data = load_data('./dataset/thresholds')
    #print(f'images data shape: {gestures_data.shape}')
    #print(f'labels data shape: {labels_data.shape}')
    #plt.imshow(gestures_data[0])
    #plt.show()

    gestures_train, gestures_test, labels_train, labels_test = train_test_split(gestures_data, labels_data, test_size= 0.1)
    file_path = './models/saved_model.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc',
                                   min_delta=0,
                                   patience=10,
                                   verbose=1,
                                   mode='auto',
                                   restore_best_weights=True)

    vgg_base = VGG16(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    optimizer1 = optimizers.Adam()

    base_model = vgg_base
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dense(128, activation='relu', name='fc3')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='fc4')(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary
    for layer in base_model.layers:
        layer.trainable = False

    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    #model.fit(gestures_train, labels_train, epochs=50, batch_size=64, validation_data=(gestures_train, labels_train), verbose=1,
    #          callbacks=[early_stopping, model_checkpoint])

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=45.,
        width_shift_range=0.3,
        height_shift_range=0.3,
        validation_split=0.1,
        horizontal_flip=False)

    datagen.fit(gestures_train)
    model.fit(datagen.flow(gestures_train, labels_train, batch_size=32, subset='training'),
                        steps_per_epoch=len(gestures_train)/32,
                        epochs=10,
                        validation_data=datagen.flow(gestures_train, labels_train, batch_size=32, subset='validation'),
                        verbose=1)


