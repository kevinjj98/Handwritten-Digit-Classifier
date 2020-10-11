'''
Small application written in Python that uses Tensorflow to train and test a neural network and, after preprocessing the 
image file to make it the proper color and dimensions for the model, predicts the individual digits from 0-9 in the image.
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# CREATING, TRAINING, AND TESTING THE NEURAL NETWORK MODEL

# Acquiring dataset of handwritten digits from Tensorflow
mnist = tf.keras.datasets.mnist

# Separating training and testing data by using the load_data function
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalizing down the data from 0-255 to 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Loading previously saved model (only if the model has already been created the first time)
# model = tf.keras.models.load_model('digits.model')

# Creating a basic neural network
model = tf.keras.models.Sequential()
# Input layer is a 1D Flatten layer
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
# 2 Dense inner layers with 128 neurons each, using the relu activation function
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=128, activation=tf.nn.relu))
# Output layer is a Dense layer with 10 neurons and softmax function
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

# Compiling the neural network
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the neural network with 3 epochs
model.fit(x_train, y_train, epochs=3)

# Evaluating the model by printing the accuracy and loss of the data
loss, accuracy = model.evaluate(x_test, y_test)
print(f'The accuracy is {accuracy}')
print(f'The loss is {loss}')

# After training the neural network once, save the results so training doesn't have to happen every time
model.save('digits.model')


# DEFINING FUNCTIONS THAT WILL BE USED TO PREPARE AN INPUT IMAGE

def x_cord_contour(contour):
    # This function takes in a contour as input and returns the x centroid coordinates
    
    if cv.contourArea(contour) > 10:
        M = cv.moments(contour)
        return (int(M['m10']/M['m00']))
    else:
        return 0

def makeSquare(not_square):
    # This function takes an image and makes the dimensions square, adding black pixels as padding if necessary
    
    BLACK = [0, 0, 0]
    img_dim = not_square.shape
    height = img_dim[0]
    width = img_dim[1]
    if (height == width):
        square = not_square
        return square
    else:
        doublesize = cv.resize(not_square, (2*width, 2*height), interpolation = cv.INTER_CUBIC)
        height = height * 2
        width = width * 2
        if (height > width):
            pad = (height - width) / 2
            doublesize_square = cv.copyMakeBorder(doublesize, 0, 0, int(pad), int(pad), cv.BORDER_CONSTANT, value=BLACK)
        else:
            pad = (width - height) / 2
            doublesize_square = cv.copyMakeBorder(doublesize, pad, pad, 0, 0, cv.BORDER_CONSTANT, value=BLACK)
    return doublesize_square

def resize_to_pixel(dimensions, image):
    # This function re-sizes an image to the specified dimensions
    
    buffer_pix = 4
    dimensions = dimensions - buffer_pix
    squared = image
    r = float(dimensions) / squared.shape[1]
    dim = (dimensions, int(squared.shape[0] * r))
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    img_dim2 = resized.shape
    height_r = img_dim2[0]
    width_r = img_dim2[1]
    BLACK = [0, 0, 0]
    if (height_r > width_r):
        resized = cv.copyMakeBorder(resized, 0, 0, 0, 1, cv.BORDER_CONSTANT, value=BLACK)
    if (height_r < width_r):
        resized = cv.copyMakeBorder(resized, 1, 0, 0, 0, cv.BORDER_CONSTANT, value=BLACK)
    p = 2
    ReSizedImg = cv.copyMakeBorder(resized, p, p, p, p, cv.BORDER_CONSTANT, value=BLACK)
    return ReSizedImg



# LOADING AN IMAGE, PREPROCESSING IT, AND CLASSIFYING THE DIGITS

image = cv.imread('assets/test.png')
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Blur the image and find edges using Canny
blurred = cv.GaussianBlur(gray, (5, 5), 0)
edged = cv.Canny(blurred, 30, 150)

# Find the contours in the image
_, contours, _ = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

# Sort out the contours from left to right using their x coordinates
contours = sorted(contours, key = x_cord_contour, reverse = False)

# An empty array will store the entire number
full_number = []

# Loop over all the contours
for c in contours:
    # Compute the bounding box for the rectangle
    (x, y, w, h) = cv.boundingRect(c)
    # Disregard tiny, irrelevant bounding boxes
    if w >= 5 and h >= 25:
        # Resize and preprocess each digit properly for the model
        roi = blurred[y:y + h, x:x + w]
        ret, roi = cv.threshold(roi, 127, 255, cv.THRESH_BINARY_INV)
        squared = makeSquare(roi)
        final = resize_to_pixel(28, squared)
        final_array = np.array([final])
        # Make a prediction on the single digit and store in the array
        prediction = model.predict(final_array)
        prediction_num = np.argmax(prediction)
        full_number.append(prediction_num)
        # Draw the bounding boxes and predicted number on the original image
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv.putText(image, str(prediction_num), (x, y + 155), cv.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0), 2)

# Display the original image with all of the bounding boxes and classified digits
plt.imshow(image)
# Convert color setting back into normal and save the image
image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
cv.imwrite('save.png', image)
print(f'DONE! The recognized numbers are: {full_number}')