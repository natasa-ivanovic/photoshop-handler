import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
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
    labels_data = np.array(labels_data)
    return gestures_data, labels_data


def load_data(directory):
    gestures_data = []
    labels_data = []
    paths = []
    for filename in os.listdir(directory):
        path = directory + '/' + filename
        paths.append((filename, path))
    random.shuffle(paths)
    for fp in paths:
        gesture_name = fp[0].split("_")[0]
        gestures_data.append(process_image(fp[1]))
        labels_data.append(gestures_map[gesture_name])

    gestures_data, labels_data = process_data(gestures_data, labels_data)
    return gestures_data, labels_data


if __name__ == '__main__':
    gestures_data, labels_data = load_data('./dataset/thresholds')

    gestures_train, gestures_test, labels_train, labels_test = train_test_split(gestures_data, labels_data, test_size= 0.1)
    file_path = './models/mobilenet/saved_model.hdf5'
    model_checkpoint = ModelCheckpoint(filepath=file_path, save_best_only=True, verbose=1, monitor='val_accuracy')
    early_stopping = EarlyStopping(monitor='val_accuracy',
                                   min_delta=0,
                                   patience=20,
                                   verbose=1,
                                   mode='auto',
                                   restore_best_weights=True)

    vgg_base = MobileNetV3Small(include_top=False, input_shape=(image_size, image_size, 3))
    optimizer1 = optimizers.Adam()

    base_model = vgg_base
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(5, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.summary()

    callback_list = [model_checkpoint, early_stopping]

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

    preprocess_input(gestures_train)

    number_training_el = int(0.9*len(gestures_train))
    gestures_validation = gestures_train[number_training_el:]
    gestures_train = gestures_train[:number_training_el]
    labels_validation = labels_train[number_training_el:]
    labels_train = labels_train[:number_training_el]

    history = model.fit(gestures_train, to_categorical(labels_train), batch_size= 64, steps_per_epoch=len(gestures_train)/64,
              epochs=100,
              validation_data=(gestures_validation, to_categorical(labels_validation)),
              verbose=1,
              callbacks=callback_list)

    total = labels_test.size

    predictions = model.predict(gestures_test)
    predictions_max = np.array(np.argmax(predictions, axis=1))

    print(predictions)
    print(predictions_max)

    check = predictions_max == labels_test
    unique, counts = np.unique(check, return_counts=True)
    result_dict = dict(zip(unique, counts))
    correct = result_dict[True]
    print(result_dict)

    print("Result: ", correct, "/", total, " correct")
    print("Accuracy: ", correct/total*100, "%")

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy - MobileNetSmall')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
