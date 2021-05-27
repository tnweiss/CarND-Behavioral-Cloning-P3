import pickle
import matplotlib.pyplot as plt

DATA_DIR = '../data'
LOG_FILE = '../data/driving_log.csv'
BATCH_SIZE = 35
HISTORY_PICKLE = '../lossHistory.pkl'
MODEL_OUTPUT = '../model.h5'


def run_show_loss():
    # load the history object
    history_object = pickle.load(open(HISTORY_PICKLE, 'rb'))
    
    # graph the data
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.show()

    
if __name__ == '__main__':
    run_show_loss()
