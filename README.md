[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/steps.png "Steps"
[image5]: ./images/CM.png "Con Matrix"


## Project Overview

Most people currently has pet as part of their family members a higth portion are Dogs. The number of 
dog-related incidents of injury are constatly increasing some of this situation are happening unawering of 
the owner making it difficult to identify dog breeds. This leads to a need for dog identification using 
modern visual technology, both for dog recognition and finer-grained
classification to breed.

The idea behind is to develop an app that given an image of a dog, the algorithm will identify an estimate of the canine’s breed using Convolutional Neural Networks (CNN)!. If supplied an image of a human, the code will identify the resembling dog breed by building a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.

![Sample Output][image1]


## Project Statemet

This is a supervised classification machine learning model where I will use state-of-the-art CNN models for 
a dog classification, the goal is piece together a series of CNN models designed to perform various tasks in 
a data processing pipeline as a result it will be possible to estimate the canine’s breed. 


### Datasets and Inputs

The dataset is clean and provided by udacity it containg 13233 total human images and 8351 total dog images. The following links are the data provided.

2. https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). The folder should contain 133 folders, each corresponding to a different dog breed.
3. http://vis-www.cs.umass.edu/lfw/lfw.tgz).  human dataset. 

### Solution statement

Steps to achieve this process:

	Step 0: Import Datasets
	Step 1: Detect Humans
	Step 2: Detect Dogs
	Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
	Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
	Step 5: Write the Algorithm
	Step 6: Test the Algorithm


The image clasificationes steps

![Sample Output][image4]

The process:

Download and visualize the data; after this step is complete the data must be normalized for further processing by the layers of the neural network. I will be using one of the openCv pre-trained face detectors, to build the human face recognition model and build a human face detector. Then i will use the Pre-trained VGG-16 Model, along with weights that have been trained on ImageNet, a very large, very popular dataset used for image classification and other vision tasks followed by building the dog detector. After we have functions for detecting humans and dogs in images, we need a way to predict breed from images. The following step involve create a CNN that classifies dog breeds, from scratch, we will target per a test accuracy of at least 10%. I'll use transfer learning to create a CNN that attains greatly improved accuracy. The final steps are to train, predict and optimize the model. 




### A benchmark model

I will use as benchmark model the article "A new dataset of dog breed images and a benchmark for finegrained 
classification by Ding-Nan Zou1,2, Song-Hai Zhang1 (), Tai-Jiang Mu1, and Min Zhang". Where they achieved an 
accuracy of 82.65% of the model trained on Tsinghua Dogs achieved.


### Evaluation metrics
Confusion Matrix
Evaluation of the performance of a classification model is based on the counts of test records correctly and incorrectly predicted by the model. The confusion matrix provides a more insightful picture which is not only the performance of a predictive model, but also which classes are being predicted correctly and incorrectly, and what type of errors are being made. To illustrate, we can see how the 4 classification metrics are calculated (TP, FP, FN, TN), and our predicted value compared to the actual value in a confusion matrix is clearly presented in the below confusion matrix table.
![Sample Output][image5]

### Project design
