# Behavioral Cloning

## Required Files

### Are all required files submitted?

<b> The submission includes a model.py file, drive.py, model.h5 a writeup report and video.mp4.</b>

- `model.py` has code to build a model, train the model, save the loss history, and save the model
- `model.h5` the model created by `python model.py`
- `video.mp4` the video output from my autonomous run

## Quality of Code

### Is the code functional?

<b> The model provided can be used to successfully operate the simulation. </b>

Run `python drive.py model.h5` to start the server used by the Udacity application to calculate steering angles given an image.

### Is the code usable and readable?

<b> The code in model.py uses a Python generator, if needed, to generate data for training rather than storing the training data in memory. The model.py code is clearly organized and comments are included where needed. </b>

In the `project` package you will find `data.py` and `model.py`. These two modules provide helper funcitons to load data, generate models, and load batches of data to help reduce memory consumption.
`run_create_model.py` and `run_show_loss.py` are modules that can be used to test some of the helper funcitons in `data.py` and `model.py`.

## Model Architecture and Training Strategy

### Has an appropriate model architecture been employed for the task?

<b> The neural network uses convolution layers with appropriate filter sizes. Layers exist to introduce nonlinearity into the model. The data is normalized in the model. </b>

Here we normalize the incoming data, by mean centering the pixel values around 0 and cropping the image to remove things like the sky and hood of the vehicle. This allows the model
to focus on things that pertain more towards the steering angle.

```python
model.add(Cropping2D(cropping=((65,25), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5))
```

After normalization we have the convolutional layers based on the network structure provided in this nvidia blog
https://developer.nvidia.com/blog/deep-learning-self-driving-cars

```python
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
```

I then flatten the values and provide a dropout layer to reduce overfitting.

```python
model.add(Flatten()) 
model.add(Dropout(.1))
```

Then we provide 4 fully connected layers that gradually scale down to our output of 1

```python
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
    
### Has an attempt been made to reduce overfitting of the model?

<b> Train/validation/test splits have been used, and the model uses dropout layers or other methods to reduce overfitting. </b>

I decided to use a 80/20 split for training/validation sets. As seen before I also include a dropout layer.

```python
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
```

### Have the model parameters been tuned appropriately?

<b> Learning rate parameters are chosen with explanation, or an Adam optimizer is used. </b>

I decided to use the Adam optimizer for training

```python
model.compile(loss='mse', optimizer='adam')
```

### Is the training data chosen appropriately?

<b> Training data has been chosen to induce the desired behavior in the simulation (i.e. keeping the car on the track). </b>

I collected the following data for my model

- 2 full runs around the track driving down the center of the road
- 1 full run around the track making corrections for a car that is too far on the right
- 1 full run around the track making corrections for a car that is too far on the left

## Architecture and Training Documentation

### Is the solution design documented?

<b> The README thoroughly discusses the approach taken for deriving and designing a model architecture fit for solving the given problem. </b>

?????

### Is the model architecture documented?

<b> The README provides sufficient details of the characteristics and qualities of the architecture, such as the type of model used, the number of layers, the size of each layer. Visualizations emphasizing particular qualities of the architecture are encouraged. </b>

??????

### Is the creation of the training dataset and training process documented?

<b> The README describes how the model was trained and what the characteristics of the dataset are. Information such as how the dataset was generated and examples of images from the dataset must be included. </b>

???????

### Is the car able to navigate correctly on test data?

<b> No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle). </b>

???????
