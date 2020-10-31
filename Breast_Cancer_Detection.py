# -*- coding: utf-8 -*-
from google.colab import files
from sklearn.utils import compute_class_weight
import cv2
import gc

import os
import re
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing.image import img_to_array, load_img
from keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import tensorflow as tf
from google.colab import drive

drive.mount('/content/drive')


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                         count=int(width)*int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def read_csv(filename):
    metadata = list()
    with open(filename) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            metadata.append(row)
    return metadata


if __name__ == "__main__":
    imageSize = 512
    ''' Helper Map '''
    abnormalityToNum = {'NORM': 0, 'CALC': 1, 'CIRC': 2,
                        'SPIC': 3, 'MISC': 4, 'ARCH': 5, 'ASYM': 6}
    numToAbnormality = {0: 'NORM', 1: 'CALC', 2: 'CIRC',
                        3: 'SPIC', 4: 'MISC', 5: 'ARCH', 6: 'ASYM'}

    ''' Metadata Pre-processing'''
    csvPath = "/content/drive/My Drive/metadata.csv"
    metadata = read_csv(csvPath)
    labels = set()
    for data in metadata:
        labels.add((data[0], data[2]))
    labels = [i[1] for i in sorted(labels)]
    train_labels = [abnormalityToNum[i] for i in labels]

    ''' Image Pre-processing '''
    pgmPath = "/content/drive/My Drive/MIAS Database/"
    pgmNames = sorted(os.listdir(pgmPath))
    images = list()
    for pgm in pgmNames:
        image = read_pgm(str(pgmPath+pgm))
        images.append(cv2.resize(image, (imageSize, imageSize)))
    images = np.asarray(images)
    train_img = images / 255.0
    train_img = train_img.reshape((-1, imageSize, imageSize, 1))

classWeight = compute_class_weight(
    'balanced', np.unique(train_labels), train_labels)
classWeight = dict(enumerate(classWeight))

model = keras.Sequential(
    [keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(imageSize, imageSize, 1)),
     keras.layers.MaxPooling2D((2, 2)),
     keras.layers.Conv2D(64, (3, 3), activation='relu'),
     keras.layers.MaxPooling2D((2, 2)),
     keras.layers.Conv2D(128, (3, 3), activation='relu'),
     keras.layers.MaxPooling2D((2, 2)),
     keras.layers.Conv2D(128, (3, 3), activation='relu'),
     keras.layers.Flatten(),
     keras.layers.Dropout(0.5),
     keras.layers.Dense(512, activation='relu'),
     keras.layers.Dense(256, activation='relu'),
     keras.layers.Dense(7, activation='softmax')]
)
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(np.array(train_img), np.array(train_labels),
          epochs=20, class_weight=classWeight)


abnormToDef = {"NORM": "Normal", "CALC": "Calcification", "CIRC": " Well-defined/circumscribed masses",
               "SPIC": "Spiculated masses", "MISC": "ill-defined masses", "ARCH": "Architectural distortion", "ASYM": "Asymmetry"}
demo = plt.figure(figsize=(40, 40))
start = 204
for i in range(start, start + 8):
    k = i - start
    ax = plt.subplot(4, 4, k+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    prediction = np.argmax(model.predict(np.expand_dims(train_img[i], 0)))
    if(numToAbnormality[prediction] == "NORM"):
        plt.imshow(np.squeeze(train_img[i]))  # Greens
    else:
        plt.imshow(np.squeeze(train_img[i]))  # Reds
    if(numToAbnormality[prediction] == "NORM"):
        norm = plt.xlabel(
            abnormToDef[numToAbnormality[prediction]], fontsize=35).set_color("green")
    else:
        abnorm = plt.xlabel(
            "Cancer: " + abnormToDef[numToAbnormality[prediction]], fontsize=35).set_color("red")

plt.show()
