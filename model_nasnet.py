import os
import random
import keras
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.nasnet import preprocess_input
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
    paths = []
    for filename in os.listdir(directory):
        path = directory + '/' + filename
        paths.append((filename, path))
    random.shuffle(paths)
    # fp = (filename, path)
    for fp in paths:
        gesture_name = fp[0].split("_")[0]
        gestures_data.append(process_image(fp[1]))
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
    file_path = './models/nasnet/saved_model.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, verbose=1, monitor='val_accuracy')
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=15,
                                   verbose=1,
                                   mode='auto',
                                   restore_best_weights=True)

    vgg_base = NASNetMobile(weights='imagenet', include_top=False, input_shape=(image_size, image_size, 3))
    optimizer1 = optimizers.Adam()

    base_model = vgg_base
    x = base_model.output
    x = Flatten()(x)
    x = Dense(128, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu', name='fc2')(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()
    # for layer in base_model.layers:
    #     layer.trainable = False

    callback_list = [model_checkpoint]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    preprocess_input(gestures_train)

    datagen = ImageDataGenerator(
        validation_split=0.2)

    datagen.fit(gestures_train)
    history = model.fit(datagen.flow(gestures_train, labels_train, batch_size=32, subset='training'),
              epochs=100,
              validation_data=datagen.flow(gestures_train, labels_train, batch_size=32, subset='validation'),
              verbose=1,
              callbacks=callback_list)

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy - NASNET')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

