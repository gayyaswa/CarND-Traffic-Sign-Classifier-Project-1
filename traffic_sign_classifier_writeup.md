# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./data-exploration-charts/y_train_hist.jpg "Visualization"
[image2]: ./data-exploration-charts/y_valid_hist.jpg "Visualization"
[image3]: ./data-exploration-charts/y_test_hist.jpg "Visualization"
[image4]: ./traffic-signs-data-test/General_Caution.jpg "General Caution"
[image5]: ./traffic-signs-data-test/Pedestrian.jpg "Pedestrain"
[image6]: ./traffic-signs-data-test/Speed_Limit_30.jpg "Speed Limit 30"
[image7]: ./traffic-signs-data-test/Speed_Limit_30.jpg "Speed Limit 50"
[image8]: ./traffic-signs-data-test/Straight_Ahead.jpg "Straight Ahead"
[image9]: ./traffic-signs-data-test/Yield.jpg "Yield"


---
### Writeup / README

link to my [project code](https://github.com/gayyaswa/CarND-Traffic-Sign-Classifier-Project-1/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used python len functions to compute the size of the data and for computing the number of class created a set out of y_train labels and computed the length.

signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Historgram of the output labels of training, validation and test dataset. The distribution graphs helps identify classes that had less occurence in the dataset. In case of the pedestrian class the occurence of them in the training data set is very less in comparison with say speed limit dataset. On testing those classes images from web the netword couldn't classify the pedastrian image.

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I just shuffled the train images and tried training the Lenet network without doing any normalization and was able to achieve 89% accuracy. Even after increasing the epochs to a higer value of 100 and tuning other hyper parameters didn't observe any improvement in accuracy. 

So decided to zero center the input images as suggested in the class with mean and standard deviation reduced, I tried tuning down the epoch to 20 and observed a quick convergence of the values also the accuracy improved to 93%. I still don't understand all the details of numerical optimization but having smaller input values centered around 0 had a bigger impact for this project

I decided to skip the data augumentation eventhough implementing it could have helped the network classify the pedestrian image. I believe the under representation of pedestrian images in the training set be the reason for Lenet to not classify it correctly when tested using an image from web.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution filter 5x5X6     	| 1x1 stride, VALID padding, outputs 28x28x6 	|
| RELU					|		Activation									|
| dropout  | Keep Probability 0.75 |
| Max pooling	 Ksize 2X2     	| 2x2 stride,  outputs 14x14x6 				|
| Convolution filter 5x5x16	 | 1x1 stride, VALID padding, output 10X10X16 |      									|
| RELU					|		Activation									|
| dropout  | Keep Probability 0.75 |
| Max pooling	 Ksize 2X2     	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		Input 400| output 120				|
| RELU					|		Activation									|
| dropout  | Keep Probability 0.75 |
| Fully connected		Input 120| output 84				|
| RELU					|		Activation									|
| dropout  | Keep Probability 0.75 |
| Fully connected		Input 84| output 10				|
| RELU					|		Activation									|
| dropout  | Keep Probability 0.75 |
| Softmax				|         									|
|			Adam Optimizer			|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.


I adapted the Lenet architecture model from the class for traffic classification:

* Type of Optmizer: AdamOptimizer
* Batch Size: 128
* Number of epochs: 200
* learning rate: 0.001
* mu: 0.1
* dropout keep probablity: 0.75

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.963
* test set accuracy of 0.961

The major steps which helped in achieving the set project goal accuracy of 0.93 are highlighted below:

* I adapted the Lenet architecture from the class and modfied the convolution filter size and max pooling size to accomodate the traffic test input image size 32x32x3 and without any other modification was able to get validation accuracy of 0.89.In the following iterative fashion tried improving the accuracy:
* Increased the epoch to 100 and observed no improvement in the accuracy so decided to try normailzing the image as suggested in the class
* Zero centered the input images (px - 128/128) as suggested in the class with mean and standard deviation reduced was able to observe a quick convergence of the values also the accuracy improved to 93%. I believe the Lenet was sensitive to varying intensities of the image by normalizing the image helped the network accuracy to improve
* As suggested in the class tried adding the dropout layer to the network so that model wouldn't overfit for the trained data. I did observe adding the dropout function for fully connected layer helped imrproving the accuracy to 96%. This make sense as the number of parameters in fully conencted layer is higher than the convolution layer and those layer might have started overfitting for the input data
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

 ![image4] | ![image5]
 --------- | ------------
 ![image6] | ![image7] 
![image8] | ![image9]

The second pedestrian image was difficult to identify as it was under represented in the training set. Looking at the distributed graph of the classes confirms this.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| General Caution      		| General Caution   									| 
| Pedestrian     			| Right of way at the next intersection 										|
| Speed Limit 30 km/h					| Speed Limit 30 Km/h											|
| Speed Limit 50 km/h	      		| Speed Limit 50 Km/h
| Ahead only			| Ahead only			|
| Yield | Yield |


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 0.83.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is sure that this is a General Caution (probability of 1.0), and the image does contain a General Caution sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1         			| General Caution   									| 
| 0     				| Speed Limit 20km/h 										|
| 0					| Speed Limit 30km/h											|
| 0	      			| Speed Limit 50km/h				 				|
| 0				    | Speed Limit 60km/h      							|


For the second image, the model is sure that this is a Right of way at next intersection (probability of 0.89) and the image is a Pedestrian. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.89         			| Right of way at next intersection   									| 
| 0.07     				| Children crossing							|
| 0.009					| Bicycles crossing											|
| 0.009	      			| Beware of ice/snow				 				|0
| 0.007				    | Turn Left Ahead      							|

For the third image, the model is relatively sure that this is a Speed Limit 30 Km/h (probability of 0.9), and the image does contain a Speed Limit 30 Km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| Speed Limit 30 Km/h   									| 
| 0.009     				| Speed Limit 50km/h 										|
| 0.0005					| Speed Limit 70km/h											|
| 0.0002	      			| Speed Limit 60km/h				 				|
| 0.00003				    | Speed Limit 20km/h      							|

For the fourth image, the model is sure that this is a Speed Limit 50 Km/h (probability of 0.99), and the image does contain a Speed Limit 50 Km/h. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| Speed Limit 50 Km/h   									| 
| 0.0000009     				| Speed Limit 50km/h 										|
| 0.00000007					| Speed Limit 70km/h											|
| 0.000000006	      			| Speed Limit 60km/h				 				|
| 0.0000000008				    | Speed Limit 20km/h      							|

For the fifth image, the model is sure that this is a Ahead only (probability of 0.99), and the image does contain a Ahead only. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99         			| Ahead Only   									| 
| 0.003     				| Turn left ahead 										|
| 0.0001					| Children crossing											|
| 0.0000008	      			| Go Straight or right				 				|
| 0.0000006				    | Roundabout mandatory      							|

For the sixth image, the model is sure that this is Yield (probability of 1.0), and the image does contain Yield. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield   									| 
| 5.47e-23      				| Speed Limit 30 Km/h 										|
| 4.82e-25					| Stop											|
| 1.71e-26	      			| Speed	Limit 80 Km/h			 				|
| 1.71e-29				    | Right-of-way at the next intersection      							|




### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


