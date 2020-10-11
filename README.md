# Handwritten-Digit-Classifier
Small application written in Python. Uses OpenCV and Tensorflow to train and test a neural network to recognize handwritten digits.

The neural network is created and trained with the MNIST dataset. It has 60,000 samples, which will help the model become very accurate. Testing the model results in an accuracy rate of over 95% most of the time.
![Training the neural network](readme-imgs/training.PNG)

Here is the original sample image that the model uses to predict the digits of once it's trained.
![Original image](readme-imgs/original.png)

After preprocessing the sample image, the code is able to use OpenCV to break apart each individual contour within the image.
![All individual contours](readme-imgs/plots.PNG)

Every individual digit contour is converted into a 28x28 square image and fed into the neural network model to make a prediction.
![Model makes a prediction](readme-imgs/predictions.PNG)

The predicted numbers along with their corresponding bounding boxes are also drawn on the original image and saved to the folder.
![Digits and bounding boxes](readme-imgs/boxes-and-digits.png)
