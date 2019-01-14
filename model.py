
# coding: utf-8


import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adam
from keras.models import model_from_json
from keras.utils import np_utils
from keras.preprocessing import image
from keras import backend as K


# Convert CSV file into pandas dataframe 
# Create separate frames for center, left and right images with corresponding angles
data = pd.read_csv('ud-data/driving_log.csv')
data_left = data.copy()[['left', 'steering']]
# Add correction factor for the left and right cameras to simulate recovery
correction = 0.15
data_left['steering'] = data_left['steering'] + correction

data_right = data.copy()[['right', 'steering']]
data_right['steering'] = data_right['steering'] - correction

data_center = data.copy()[['center', 'steering']]

data_left.rename(columns={'left': 'image'}, inplace=True)
data_right.rename(columns={'right': 'image'}, inplace=True)
data_center.rename(columns={'center': 'image'}, inplace=True)

# Combine into one frame with images and steering angles
comb_data = pd.concat([data_center, data_left, data_right])

flip_data = comb_data.copy()
comb_data['flip'] = False
flip_data['flip'] = True
flip_data['steering'] = flip_data['steering'] * -1.0

final_data = pd.concat([comb_data, flip_data])


# Crops image to include on area of interest (discards bumper and unwanted scene)
def crop(image, top=65, bottom=140):
    return image[top:bottom]

# function to resize images to acceptable size
def resize(image):
    return cv2.resize(image,(128, 128), interpolation= cv2.INTER_AREA)

# increate or decrease the brightness of image based on uniform probibility
def augment_brightness(image):
    bright = np.random.uniform(0.4, 1.2)
    
    # Convert to HSV and modify V channel
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv[:,:,2] = hsv[:,:,2] * bright
    
    # Convert from HSV back to RGB
    chng_bright = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return chng_bright

#Randomly deciding to change brightness
def random_bright_augment(image, angle):
    choice = np.random.randint(2)
    if choice == 1:
        return augment_brightness(image), angle
    else:
        return image, angle

# preprocessing images: cropping, resizing, changing brightness and flipping
def augment_data(entry):
    file = 'ud-data/'
    flip, steer_angle = entry
    image = plt.imread(file+flip[1]['image'].strip())
    steer_angle = steer_angle[1]['steering']
    image = resize(crop(image))
    image, steer_angle = random_bright_augment(image, steer_angle)
    
    flip = flip[1]['flip']
    if flip:
        image = image[:,:,::-1]

    return image, steer_angle


images = final_data[['image', 'flip']]
angles = final_data[['steering']]


# split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.1)


# generator to generate aygmented data on the fly to avoid running out of memory

def generator(X, y, batch_size=128):
    N = X.shape[0]
    total_batches = int(np.ceil(N / batch_size))
    while True:
        X, y  = shuffle(X, y)
        for i in range(total_batches):
            start = i*batch_size
            end = (i+1)*(batch_size)
            if end <= N:
                X_batch = X[start:end]
                y_batch = y[start:end]
            else:
                X_batch = X[start:]
                y_batch = y[start:]
            
            X_batch, y_batch = X_batch.iterrows(), y_batch.iterrows()
            X_image_batch, y_batch = zip(*map(augment_data, zip(X_batch, y_batch)))
            X_image_batch = np.asarray(X_image_batch)
            y_batch = np.asarray(y_batch)
            yield X_image_batch, y_batch
    

# create generators to create training and validation batches
training_generator = generator(X_train, y_train)
validation_generator = generator(X_val, y_val)


# NVIDIA model + some added dropout layers through experimentation to prevent overfitting
model = Sequential([
        Lambda(lambda x: (x/ 127.5 - 1.0),input_shape=(128,128,3)),
        Convolution2D(24,5,5, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Convolution2D(36,5,5, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.5),
        Convolution2D(48,5,5, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.2),
        Convolution2D(64,3,3, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Convolution2D(64,3,3, border_mode='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dropout(0.3),
        Dense(500, activation='relu'),
        Dropout(0.4),
        Dense(100, activation='relu'),
        Dropout(0.2),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1),
    ])

model.compile(optimizer=Adam(), loss='mse')

print(model.summary())

model.fit_generator(training_generator, X_train.shape[0], nb_epoch=8, validation_data=validation_generator, nb_val_samples=X_val.shape[0])


model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("Model trained and saved")





