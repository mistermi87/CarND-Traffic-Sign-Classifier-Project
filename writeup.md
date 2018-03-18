# **Traffic Sign Recognition** 

## Writeup


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image1]: ./Resources/X_new.png "New traffic signs that were not in the testing set (Photos from german Streets by me)"
[image2]: ./Resources/Visualization.png "Visualization"
[image3]: ./Resources/relu1.png "Traffic Sign 6 [Turn Left only ahead] after 1st Layer activation"
[image4]: ./Resources/relu2.png "Traffic Sign 6 [Turn Left only ahead] after 2nd Layer activation"
[image5]: ./Resources/Dataset-Classes.png "Histogram of Dataset Classes"
[image6]: ./Resources/X_train.png "Random Training data"
[image7]: ./Resources/X_train_corr.png "Random Training data corrected"
[image8]: ./Resources/Error.png "Validation-Error over Epochs"
## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This is the writeup. I have to versions of the project:
* Version 1: 660000 Trainable Variables and a testing accuracy of 95,9 [project code](./Traffic_Sign_Classifier-660K.ipynb)
* Version 2: 800000 Trainable Variables and a testing accuracy of 96,3[project code](./Traffic_Sign_Classifier-800K.ipynb)

The second version is more accurate, but because I want to use this code on a Raspberry Pi, I tried to keep the model as small as possible. That is why I kept the first version.
This writeup focuses on the first version.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the array.shape() function to retrieve the Shape of the:
I order to get the bumber of classes I calculated the difference of the maximum label number and the minimum of training/validation/testing datasets. A plot of the histogram confirmed that there are no empty classes

* The overal size of the training data is (34799, 32, 32, 3)
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a distribution of the different classes over the whole data:

![Histrogram][image5]

It is possible to see that the there is a different amount of data available for each class, but that the overal distribition between training/validation/testing is roughly 0.67/0.09/0.24 in each class.

Here are examples from the training data
![Example images][image6]
It becomes apparent, that image normalization is necessary, as some Images have a really low contrast or are really dark.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

### Image Preprocessing and Augumentation:
#### Augmentation
##### How:
I copied the training data three times and concatenated it to a long array of ~140000 images. The copied images were altered as follows: 
* the first copied part was rotated randomly with up to +/- 20 deg. In order to avoid white corners in the rotated images, the images were scaled appropriately
* the second copied part was just scaled randomly up to a factor of 1/0.75. There was no shifting of the center of the image
	> I tried to use the tensorflow functions of  tf.contrib.image. ... to accelarate the processes by using my GPU for these operations, but it seems that it these functions are not yet supported in Windows 10 in our tensorflow version [Github Issue](https://github.com/tensorflow/tensorflow/issues/9672)
* the last copied part was altered by overlaying a random shadow on the image. This was done by darkening the image on one side of a line that started from a random point on the upper edge of the image and ended at a random point on the lower edge of the image.
	> This code was not mine but from [here](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9). I altered it by making it available for all input image sizes and omitting the RGB-->HLS-->RGB-conversion as it does not work well with aliasing artifacts in the small images and will give strong rainbow edges. Instead of manipulating the "L"(Lightness) dimension of the HLS-space I just multiplied each RGB-channel by 0.75 in the "shaded" part

##### Why:
I realized that my model had a tendency to overfit the training data. After introducing drop-out (first with keep_rate=0.7 then with 0.5) and making it shallower I tried data augmentation. This was successful, because with each new set of augmented images the Validatition-error went down >0.2%.

#### Normalizing

I normalized all images (Train & Augmentation, Valdiation, Testing)  to values between 0 and 1. Additionally I applied contrast stretching (taking into account the maximum and minimum of all channels, so that no information is lost by clipping in R,B or G and no color shift takes place)  I did not reduce to grayscale because especially in signs the color information of a blue vs a read circle might be crucial information. In the [provided Paper in the Notebook](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) it is also possible to see, that small neural networks perform better with the RGB-Version of the Traffic-Sign Dataset

Here are two of the random examples from above: original, randomly rotated, randomly zoomed and with random shadow. All of those images are normalized.
![Augumented and normlized Example images][image7]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My first motivation was to find a model that was lightweight and small enough to do very fast forward propagation. I hope to run some of this code on a Raspberry Pi 3 some day, as I want to apply my knowledge gained in this course on building a small smart robot.
Initially I wanted to limit the amount of trainable variables in the model to 500000. This model now is a compromise with 660000 variables. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x18 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x18 				|
| Dropout		      	| keep_prob=0.7, only during training 			|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x48   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x48 					|
| Dropout		      	| keep_prob=0.7, only during training 			|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 1x1x512 *	|
| RELU					|												|
| Dropout		      	| keep_prob=0.7, only during training			|
| Flatten 		      	| 												|
| Fully connected		| Input: 512 Output: 43 **       					|
| Softmax				| (this is calculated outside of the main model) |

  \* (mathematically identical with fully connect layer)
  \** (an additional FC-layer reducing the 512 features to 256 before this layer was also tested with 10% slightly higher results, increasing the model size by 20%)

![Visualization of the model][image2]


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

* **Optimizer**: To train the model I used the Adam Optimizer.  This is one of the most efficient optimizers that also makes use of minibatches. It might lack the robustness of a simple RMS-Prop Solver but since this is a relatively simple model (in today's terms)  we need not to worry about it.
* **Batch-size**: I started with a batch size of 128 but found out that computations accelerated (Locally: i7-4700mq, Geforce GTX-765m, 8Gb Ram) with a batch size of 256
* **Epochs**: I let the model run for 100 Epochs. The final validation accuracy was already found after 40-60 Epochs (Depending on drop-out rate) , increased 0,3% temporarily to settle to the final result. This might be a sign of slight overfitting of the training-data. This effect was even more pronounced when i did not use augmented data. It might be feasible to introduce early stopping here.

![Validation-Error for each Epoch][image8]

* **Hyperparameters**: Once the final model was set I tried different learning rates. This was done by starting at a learning rate of 0.001 and decreasing the rate by a factor of $\sqrt{10}$ every 20 epochs, so that by Epoch 40 a learning rate of 0.0001 was active etc. . This did not change to results noticeably. A learning rate greater than 0.001 resulted in instability, so that the validation accuracy was a constant 0.054 or 0.048.

**Training time** was roughly 50min for 100 Epochs on the whole training set of ~140000 images including the augmented versions.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 1.000
* validation set accuracy of ~0.970 
* test set accuracy of 0.959@660K variables /  0.963@800K variables

My first approach was taking the LeNet-5 Model of the previous exercise and scaling each layer by a factor of 3, to accommodate the additional information in the three RGB-Channels of the input image compared to the single gray-scale Channel of the original Numbers-Dataset. I left all the other parameters untouched and got a testing accuracy of exactly 0.93. This would have been sufficient to pass. After that I inflated the model by making it even bigger and introducing an additional Fully connected layer (1024x1024!). I was aware that this model was very prone to overfitting the test data as it was very big compared to the problem, so it introduced dropout during training (Keep prob=0.7). With this I achieved a testing accuracy of 0.955 but the model had 2.4 Million trainable variables and was really slow.

After a bit of research I found a completely different architecture, MobileNets by Google,  that was promising on these kind of data. After 10 days of trying to implement it, I gave up. I may come back to it when I learn a bit more about Tensorflow and neural networks. What I created until then, is listed at the end of this paragraph.

After that I went back to the basic LeNet model and looked up what was done in research of CNN after 1998, when it was published, in order to improve the architecture while keeping the model size reasonably small. Interestingly until GoogleNet/Inception(2014) the basic design principles remained untouched and the models only became deeper or parallel [--> overview](https://medium.com/@siddharthdas_32104/cnns-architectures-lenet-alexnet-vgg-googlenet-resnet-and-more-666091488df5). Every model basically consisted of Blocks with [Conv1 (--> Conv2) --> Pooling] and Fully connected layers at the end. After my failing of implementing the MobileNets-architecture I was hesitant to use inception modules, residual connections or other "fancy stuff".

So I started from my 3x inflated LeNet model described above, in which I slightly decreased the sizes of the convolutional layers. I realized that I still got a training accuracy of 100%, which is an indicator of overfitting. 
This is why I introduced data augmentation and deleted one fully connected layer. Now I ended up with four layers: two convolutional layers and two fully connected layers (one of the fc-layers was realized by making a conv-layer with a kernel size similar to the width/depth of the input tensor, stride=1 and no padding, which should be mathematically equivalent). My model size shrunk from 0.8 million variables to 0.66 million. Still the model tended to overfit. That is why I introduced a stronger keep_prob of 0.5 from previously 0.7. With this I got a testing accuracy of 0.959

Using the keep_prob of 0.5, augmented data and the model with 0.8 Million variables, I archived a testing accuracy of 0.963.
It is a personal choice if a decrease off 10% in the testing error justifies an increase of 20% of the model size.

I never touched the **activation function.** [This article]( https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f) suggests the ReLU function seem as the best option unless you have a lot of  "dead nodes" --> nodes without activation, then you should give Leaky ReLUs or Maxout a try. In the last chapter where I plotted the first two convolutional layers, it is possible to see that only one of the 18 first-layer nodes and seven of the 48 second-layer nodes have no activation at all. With a different input they might be activated.

I also did not change the **max-pooling** to average-pooling as  I feared that this would have decreased performance on edge detection or working with the relatively sparse nodes of the later layers.

#### Summary:
I used the basic principles of the LeNet-5 Architecture by first inflating it and then gradually fighting overfitting by using data augmetation, Dropout with increasing rates and deleting one Fully connected layer from the original LeNet architecture. With this i was able to achive good results considering the size of my Neural Network and the "old-school" network architecture.
I learned a lot doing research on the MobileNets but was not able to implement it, yet.

In the future I would maybe make a further investigation of the model's performance by looking at certain classes and its corresponding precision and recall.

----------


#### Comment: First unsuccessfull new model: MobileNets by Google
Being motivated to implement a rather small and fast (forward-prop) CNN, I tried to apply the model-architecture described in [Howard, Andrew G., et al. "Mobilenets: Efficient convolutional neural networks for mobile vision applications." *arXiv preprint arXiv:1704.04861* (2017).](https://arxiv.org/abs/1704.04861) . In [Chen, Hong-Yen, and Chung-Yen Su. "The Enhanced Hybrid MobileNet." *arXiv preprint arXiv:1712.04698* (2017)](https://arxiv.org/abs/1712.04698) the authors apply the main idea of the MobileNets onto the CIFAR-10 and CIFAR-100 Datasets with remarkable results. Those datasets also consist of RGB-Images with a size of 32x32 each.
> The basic idea of this approach is the replacement of a regular convolutional layers by a depthwise convolution followed by a pointwise convolution with kernel size 1x1 (--> depthwise separable). This reduces the number of trainable variables manifold while loosing only little accuracy. To make up for the loss in accuracy the model is made deeper. For stability of this deep model, batch normalization is implemented after every convolutional layer.

Unfortunately I was not able to implement the approach until now. Tensorflow does not give me an error but the validation-accuracy stays at around 0.5%. This might have different reasons:

* I chose the wrong optimizer: Neither AdamOptimizer nor RMSProp work.
* I chose the wrong learning rate: I tried to lower $\alpha$ several magnitudes but it did not change the outcome
* The other hyperparameters of the Optimizer need tuning as well. I did not change the Momentum or Decay.
* Batch-normalization is not implemented correctly. As Batch-Norm has trainable parameters (mean, variance...) that are not updated by the back-propagation, I tried to train them as described [here](http://ruishu.io/2016/12/27/batchnorm/) and [here](https://www.tensorflow.org/versions/master/api_docs/python/tf/contrib/layers/batch_norm)
* Fractional Max Pooling is not working correctly

 I left the model in the most recent work-in-progress state in the Notebook provided.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

 I went around in my neighborhood in Munich/Germany and took eight usable photos of different street signs with my smartphone at around dusk. I chose poor photo conditions on purpose. Some of the images were taken from far away (noise, sharpness) and the some signs were located in the corner of the picture to produce poor quality (lens distortion, chromatic aberration) . Then I cut out the street signs and reduced those images to 32x32 Pixel in RGB. Additionally I manualy distorted some of the images.
![Testing my own images][image1]

 0. This Image is a straightforward "Stop"-sign. I was not sure how the model would handle the little "Bikes excepted" Sign on the bottom.
 1. I rotated this "Yield"-sign quite heavily, addionaly it was cropped so that the corners were cut.
 2. This "Road -work ahead"-sign was orignially between the trees. I Photoshopped the right tree in front of the sign, so that a greater percentage of the sign is covered.
 3. This "50km/h max."-sign was skewed to one side artificially.
 4. For the human eye straightforward "50km/h max."-sign. Surprisingly many algorithm iteration tested on this image struggled to find a correct result. Often times this would be classified as "*80*km/h max." Maybe the geometric shapes in the background (window frames) threw it off?
 5. This sign should also be easy to classify. In the lower part of the image a part of a red "Yield"-Sign is visible. Let's see if anything red is more probable in the Top-5 classification.
 6. Here the bike in the sign is only very small and difficult to read even for the human eye.  Maybe the classifier has problems reading this. 
 7. A straightforward "Yield"-sign no difficulties to expect here.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction of the smaller model. The bigger model hat 100% accuracy and was 100% certain for each result:

| Image			        |     Prediction	        	|  Correct
|:---------------------:|:-----------------------------:| :------------------------:|
| No entry      		| No entry  					| x
| Yield     			| Yield 						| x
| Road work				| Road work						| x
| 50 km/h	      		| Turn Left ahead				|
| 50 km/h	      		| 50 km/h						| x
| Turn Right ahead		| Turn Right ahead     			| x
| Bicycle Crossing		| Bicycle Crossing     			| x
| Yield					| Yield			      			| x


The model was able to correctly guess 7 of the 8 traffic signs, which gives an accuracy of 87.5%. This is quite poor in comparison to the test set. Maybe the way I created the data was different from the way that the Traffic Sign Dataset was obtained. This would lead to an inherent quality of the original image-set that "my" pictures do not have. Another reason could be that i did not train the model on skewed images. the miss-classified image was a skewed image.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Image: |  0 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 17 	 |  No entry
|Prediction: | | |
|1.0000 	 | ID: 17 	 |  No entry
|0.0000 	 | ID: 14 	 |  Stop
|0.0000 	 | ID: 09 	 |  No passing
|0.0000 	 | ID: 10 	 |  No passing for vehicles over 3.5 metric tons
|0.0000 	 | ID: 11 	 |  Right-of-way at the next intersection

The model was very certain, that it was a "No entry sign" and it was correct

| Image: |  1 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 13 	 |  Yield
|Prediction: | | |
|1.0000 	 | ID: 13 	 |  Yield
|0.0000 	 | ID: 15 	 |  No vehicles
|0.0000 	 | ID: 02 	 |  Speed limit (50km/h)
|0.0000 	 | ID: 03 	 |  Speed limit (60km/h)
|0.0000 	 | ID: 34 	 |  Turn left ahead

The model was very certain, that it was a "Yield"-sign and it was correct

| Image: |  2 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 25 	 |  Road work
|Prediction: | | |
|1.0000 	 | ID: 25 	 |  Road work
|0.0000 	 | ID: 29 	 |  Bicycles crossing
|0.0000 	 | ID: 00 	 |  Speed limit (20km/h)
|0.0000 	 | ID: 01 	 |  Speed limit (30km/h)
|0.0000 	 | ID: 02 	 |  Speed limit (50km/h)

The model was very certain, that it was a "Road work"-sign and it was correct

| Image: |  3 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 02 	 |  Speed limit (50km/h)
|Prediction: | | |
|1.0000 	 | ID: 34 	 |  Turn left ahead
|0.0000 	 | ID: 40 	 |  Roundabout mandatory
|0.0000 	 | ID: 05 	 |  Speed limit (80km/h)
|0.0000 	 | ID: 01 	 |  Speed limit (30km/h)
|0.0000 	 | ID: 38 	 |  Keep right

The model was very certain, that it was a "Turn left ahead"-sign and it was **not correct**.
This is interesting. It gave a really high probability to this result. One would assume that the correct result would still be under the Top-5 results, but it is not. As said before, in my opinion,  the main reason the model failed so badly here, is the fact that is was never trained on skewed images, but had to deal with one here. It may have focused on the white part between the "50" and the Border of the image and interpreted that as a white arrow.  It is surprising, that the model does not take into account the color of the sign at all. One would assume that a red circle around a sign would be a K.O.-criteria for classifying a "blue" sign.

| Image: |  4 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 02 	 |  Speed limit (50km/h)
|Prediction: | | |
|1.0000 	 | ID: 02 	 |  Speed limit (50km/h)
|0.0000 	 | ID: 05 	 |  Speed limit (80km/h)
|0.0000 	 | ID: 01 	 |  Speed limit (30km/h)
|0.0000 	 | ID: 38 	 |  Keep right
|0.0000 	 | ID: 08 	 |  Speed limit (120km/h)

The model was very certain, that it was a "Speed-limit (50km/h)"-sign and it was correct

| Image: |  5 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 33 	 |  Turn right ahead
|Prediction: | | |
|1.0000 	 | ID: 33 	 |  Turn right ahead
|0.0000 	 | ID: 00 	 |  Speed limit (20km/h)
|0.0000 	 | ID: 01 	 |  Speed limit (30km/h)
|0.0000 	 | ID: 02 	 |  Speed limit (50km/h)
|0.0000 	 | ID: 03 	 |  Speed limit (60km/h)

The model was very certain, that it was a "Turn right ahead"-sign and it was correct

| Image: |  6 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 29 	 |  Bicycles crossing
|Prediction: | | |
|0.9983 	 | ID: 29 	 |  Bicycles crossing
|0.0017 	 | ID: 28 	 |  Children crossing
|0.0000 	 | ID: 30 	 |  Beware of ice/snow
|0.0000 	 | ID: 03 	 |  Speed limit (60km/h)
|0.0000 	 | ID: 23 	 |  Slippery road

Here, the model decided correctly to classify this image as a "Bicycles crossing"-sign. Still it was not 100% sure and had a ~0.2%  prediction it being a "Children crossing"-sign which looks rather similar, taking into account the really low resolution of the input


| Image: |  7 | |
|:------------------:|:---------------:|:--------:|
Correct  | ID: 13 	 |  Yield
|Prediction: | | |
|1.0000 	 | ID: 13 	 |  Yield
|0.0000 	 | ID: 03 	 |  Speed limit (60km/h)
|0.0000 	 | ID: 14 	 |  Stop
|0.0000 	 | ID: 09 	 |  No passing
|0.0000 	 | ID: 01 	 |  Speed limit (30km/h)

The model was very certain, that it was a "Yield"-sign and it was correct

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Here I plotted the image of the "turn right ahead"-sign in the first and second activation layer. I chose this because it is quite a complex sign but easily classified. The model needs to find a white curvature in the sign that either points right or left. Does it take into account the pointed head of the arrow? :
First Layer:
![ [Turn Right only ahead] after 1st Layer activation][image3]
It is easy to see that certain nodes focus on edges going from dark to bright or vice versa. Some nodes are trained to identify vertical edges or diagonal edges. Those diagonal edges help to highlight the arrow in some nodes.  One node activates the whole sign, maybe the model has learned to recognize blue signs here? One node is highly activated on the bend of the arrow. Some nodes highlight the white inner part of the yield sign below the actual sign, but this information is not used in the following layers.

Second Layer
![[Turn Right only ahead] after 2nd Layer activation][image4]

Here the information in the layer is already to abstract to connect to a certain characteristic of the input.
