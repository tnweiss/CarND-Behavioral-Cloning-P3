import numpy as np
import pickle
from project.data import load_csv_data, load_batch_generator
from project.model import get_model
from sklearn.model_selection import train_test_split
from math import ceil
import matplotlib.pyplot as plt

DATA_DIR = 'data'
LOG_FILE = 'data/driving_log.csv'
BATCH_SIZE = 35.0
HISTORY_PICKLE = 'lossHistory.pkl'
MODEL_OUTPUT = 'model.h5'


def model():
    # load the data
    X, Y = load_csv_data(LOG_FILE, DATA_DIR)
    
    # create the training and validation generator
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    
    # create generators for the training and validation data
    train_gen = load_batch_generator(X_train, Y_train)
    valid_gen = load_batch_generator(X_valid, Y_valid)
    
    # create the model
    model = get_model()
    
    # get the training generator
    history_object = model.fit_generator(train_gen,
            steps_per_epoch=ceil(len(X_train)/BATCH_SIZE),
            validation_data=valid_gen,
            validation_steps=ceil(len(X_valid)/BATCH_SIZE),
            epochs=4, verbose=1)
    
    # pickle the loss object
    pickle.dump(history_object, open(HISTORY_PICKLE, 'wb'))
    
    # save the trained model
    model.save(MODEL_OUTPUT)

    
if __name__ == '__main__':
    model()
