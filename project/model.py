from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, MaxPooling2D, Cropping2D, Dropout


def get_model():
    model = Sequential()
    model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: (x / 255.0) - 0.5))
    
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    
    model.add(Dropout(.1))
    
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    model.compile(loss='mse', optimizer='adam')
    
    return model
