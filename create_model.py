import os
import csv
import pickle
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D

DATA_DIR = 'data'
LOG_FILE = 'data/driving_log.csv'
IMAGES_PICKLE = 'images.pkl'
MEASUREMENTS_PICKLE = 'measurements.pkl'
MODEL_OUTPUT = 'model.h5'
STEERING_CORRECTION = 0.2


lines = []
print("Parsing CSV...", end='')
with open(LOG_FILE, 'r') as f:
    r = csv.reader(f)
    for line in r:
        lines.append(line)
print("Done")


print("Opening files...", end='')
images = []
measurements = []
for i, line in enumerate(lines[1:]):
    img_center = os.path.join(DATA_DIR, line[0])
    images.append(cv2.imread(img_center))
    measurements.append(float(line[6]))
    images.append(cv2.flip(images[-1], flipCode=1))
    measurements.append(measurements[-1] * -1)
    
    img_left = os.path.join(DATA_DIR, line[1].strip())
    images.append(cv2.imread(img_left))
    measurements.append(float(line[6]) + STEERING_CORRECTION)
    
    img_right = os.path.join(DATA_DIR, line[2].strip())
    images.append(cv2.imread(img_right))
    measurements.append(float(line[6]) - STEERING_CORRECTION)

X_train = np.array(images)
Y_train = np.array(measurements)
print("Done")

print("Initializing Model...", end='')
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))
print("Done")

print("Training Model...", end='')
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=.2, shuffle=True, nb_epoch=5)
print("Done")

print("Saving Model...", end='')
model.save(MODEL_OUTPUT)
print("Done\n\n")