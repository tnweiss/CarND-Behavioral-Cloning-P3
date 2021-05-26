import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle


def load_csv_data(log_file, data_dir, steering_correction = 0.25): 
    image_filepaths = []
    measurements = []
    
    with open(log_file, 'r') as f:
        r = csv.reader(f)
        
        next(r)  # skip the header
        
        for line in r:
            image_filepaths.append(os.path.join(data_dir, line[0]))
            measurements.append(float(line[3]))

            image_filepaths.append(os.path.join(data_dir, line[1].strip()))
            measurements.append(float(line[3]) + steering_correction)

            image_filepaths.append(os.path.join(data_dir, line[2].strip()))
            measurements.append(float(line[3]) - steering_correction)
    return image_filepaths, measurements


def load_x_data(X_data, Y_data, flip=False):
    out_x = []
    out_y = []
    
    for x, y in zip(X_data, Y_data):
        img = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
        
        out_x.append(img)
        out_y.append(y)
        
        if flip:
            out_x.append(cv2.flip(img, flipCode=1))
            out_y.append(y * -1.0)
    return np.array(out_x), np.array(out_y)

                
def load_batch_generator(X_data, Y_data, batch_size=32):
    num_samples = len(X_data)
    while 1: # Loop forever so the generator never terminates
        X_data, Y_data = shuffle(X_data, Y_data)
        
        for offset in range(0, num_samples, batch_size):
            batch_x = X_data[offset:offset+batch_size]
            batch_y = Y_data[offset:offset+batch_size]

            images = []
            angles = []
            for x, y in zip(batch_x, batch_y):
                image = cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB)
                angle = float(y)
                
                # load normal
                images.append(image)
                angles.append(angle)
                
                # load flipped
                images.append(cv2.flip(image, 1))
                angles.append(angle * -1)

            yield np.array(images), np.array(angles)
            #yield shuffle(X_train, y_train)
