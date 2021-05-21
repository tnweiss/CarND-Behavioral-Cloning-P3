import os
import csv
import pickle
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense

DATA_DIR = 'data'
LOG_FILE = 'data/driving_log.csv'
IMAGES_PICKLE = 'images.pkl'
MEASUREMENTS_PICKLE = 'measurements.pkl'
MODEL_OUTPUT = 'model.h5'


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
    measurements.append(line[6])
    
    img_left = os.path.join(DATA_DIR, line[1])
    images.append(cv2.imread(img_center))
    measurements.append(line[6])
    
    img_right = os.path.join(DATA_DIR, line[2])
    images.append(cv2.imread(img_center))
    measurements.append(line[6])
X_train = np.array(images)
Y_train = np.array(measurements)
print("Done")

print("Initializing Model...", end='')
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))
print("Done")

print("Training Model...", end='')
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=.2, shuffle=True)
print("Done")

print("Saving Model...", end='')
model.save(MODEL_OUTPUT)
print("Done\n\n")