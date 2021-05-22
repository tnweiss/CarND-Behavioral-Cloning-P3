from data import load_csv_data, load_x_data
from model import get_model

DATA_DIR = '../data'
LOG_FILE = '../data/driving_log.csv'
MODEL_OUTPUT = '../model.h5'
STEERING_CORRECTION = 0.2


def run_create_model():
    # load the data
    X_train, Y_train = load_csv_data(LOG_FILE, DATA_DIR)
    X_train, Y_train = load_x_data(X_train, Y_train)
    
    # create the model
    model = get_model()
    
    # fit the model
    model.fit(X_train, Y_train, validation_split=.2, shuffle=True, nb_epoch=4)
    
    # save the model
    model.save(MODEL_OUTPUT)


if __name__ == '__main__':
    run_create_model()
