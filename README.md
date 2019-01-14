#**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./pictures/normal.jpg "Normal Image"
[image2]: ./pictures/flipped.png "Flipped Image"
[image3]: ./pictures/change_bright.jpg "Brightness changed"
[image4]: ./pictures/resize128x128.jpg "Resized 128x128"
[image5]: ./pictures/cropped&resized.jpg "crop and resize"

---
###Files Submitted & Code Quality

References for this project: (Thanks to all!)
* Vivek Yadav's blog post on medium named "An augmentation based deep neural network approach to learn human driving behavior". This helped me greatly understanding how to tackle this problem of the project. It gave an insight about how to augment and process the images and its effect on creating a robust model.
* NVIDIA end to end deep learning paper
* Finally referred to many links by students on the slack channel and on the facebook group

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing the weights of the model
* model.json containing the model
* README.md

NOTE: I had issues with using model.h5 because of some problems with the drive.py code. The car wasn't moving at all in autonomous mode. There was some issue in the drive.py script. Aaron Brown added a PI controller too to fix the issue but still my car wasn't moving after using the updated script of drive.py. I used an older version of drive.py script used by members froms earlier cohort. So please use model.json for testing with drive.py and model.h5 in the same directory as model.json.  

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.json
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

I have used the Nvidia end to end learning model with some modifications. The model contains 5 convolutional layers with 5 fully connected layers. Modifications were employed empiricallly through experimentation. 5x5 and 3x3 filters were used in the convolution layers and the depths of sizes 24,26,48,64 were used. (code line 135 onwards)

The model includes RELU layers to introduce nonlinearity (code line 135), and the data is normalized in the model using a Keras lambda layer (code line 134). 

####2. Attempts to reduce overfitting in the model

The model has dropout layers after every hidden layer with a dropout probablity of 0.1 to 0.5. I introduced a minimal dropout in most of the layers to reduce overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 159).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The Udacity dataset was used for training.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to first use LeNet architecture to check how it performs. I then switched to using the NVIDIA model as suggested by many on the forums, and yes, it gave ery good results. I tweaked the model a bit with Maxpoling and adding dropouts as described above and it performed better. It did not happen in one go, it took a lot of experimentation. I also chose to resize the images to from 160x320 to 128x128 to reduce dataset size and training parameters while also not distorting the image very much due to resizing.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. Which indicates overfitting. Also the car zig zaged on the road.

To combat the overfitting, I applied dropout to two layers, and it  showed better results. Then I added some more dropout layers to the model with few small and medium sized dropouts ranging from 0.1-0.5.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. The first left turn and the only right turn on the track. I used the right and left camera images to simulate recovery. The right and left camera images are given a correction angle (code line 26,27,30) and fed in as center camera images. This solved the problem although the correction angles had to be experimented with. Including left and right imgaes along with dropout resulyed in smoother driving.

At the end of the process, the vehicle is able to drive autonomously around track-1 without leaving the road.

####2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture.


Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_1 (Lambda)                (None, 128, 128, 3)   0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 128, 128, 24)  1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 64, 64, 24)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 64, 64, 24)    0           maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 64, 64, 36)    21636       dropout_1[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 32, 32, 36)    0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 32, 32, 36)    0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 32, 32, 48)    43248       dropout_2[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 16, 16, 48)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 16, 16, 48)    0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 16, 16, 64)    27712       dropout_3[0][0]                  
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 8, 8, 64)      0           convolution2d_4[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 8, 8, 64)      36928       maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 4, 4, 64)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 1024)          0           maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
dropout_4 (Dropout)              (None, 1024)          0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 500)           512500      dropout_4[0][0]                  
____________________________________________________________________________________________________
dropout_5 (Dropout)              (None, 500)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 100)           50100       dropout_5[0][0]                  
____________________________________________________________________________________________________
dropout_6 (Dropout)              (None, 100)           0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 50)            5050        dropout_6[0][0]                  
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 10)            510         dense_3[0][0]                    
____________________________________________________________________________________________________
dense_5 (Dense)                  (None, 1)             11          dense_4[0][0]                    
====================================================================================================
Total params: 699,519
Trainable params: 699,519
Non-trainable params: 0




####3. Creation of the Training Set & Training Process

I used the Udacity dataset for training which includes about 8000 images from the left, center and right cameras each. So approximately 24000 images. The right and left camera serve as recovery data images.

To augment the data sat, I also horizontally flipping the images and multiplying angles by -1 as the recorded images are in anti clockwise driving, so it would be more biased towards taking right turns than left. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]
 
After the collection process, I had around 48 thousand images. I then preprocessed this data by cropping to remove unwanted features like car bumperand sky and resizing to 128x128 pixels. I also randomly changed brightness of images so that the model doesn't perform bad on badly lit scenes or very bright scenes. Mean centering and normalization was used so that the optimizer achieves faster convergence. Data was normalized in the range [1,-1]. (lambda layer. line code 134). Some of the processed images below: 

![alt text][image3]
![alt text][image4]
![alt text][image5]

I finally put 10% of the data into a validation set. 

I used this training data (approx 43K images) for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8 as evidenced by flattening of the loss. I used an adam optimizer so that manually training the learning rate wasn't necessary.
