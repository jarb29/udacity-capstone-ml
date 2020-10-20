[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"
[image4]: ./images/steps.png "VGG16 Model Figure"


## Project Overview

Most people currently has pet as part of their family members a higth portion are Dogs. The number of 
dog-related incidents of injury are constatly increasing some of this situation are happening unawering of 
the owner making it difficult to identify dog breeds. This leads to a need for dog identification using 
modern visual technology, both for dog recognition and finer-grained
classification to breed.

The idea behind is to develop an app that given an image of a dog, the algorithm will identify an estimate of the canine’s breed using Convolutional Neural Networks (CNN)!. If supplied an image of a human, the code will identify the resembling dog breed by building a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.

![Sample Output][image1]


## Project Statemet

The goal is create state-of-the-art CNN models for a dog classification. By piecing together a series of 
models designed to perform various tasks in a data processing pipeline.  

### Datasets and Inputs

2. https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). The folder should contain 133 folders, each corresponding to a different dog breed.
3. http://vis-www.cs.umass.edu/lfw/lfw.tgz).  human dataset. 

### Solution statement

Steps to achieve this process:
![Sample Output][image4]

	Step 0: Import Datasets
	Step 1: Detect Humans
	Step 2: Detect Dogs
	Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
	Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
	Step 5: Write the Algorithm
	Step 6: Test the Algorithm

### A benchmark model

I will use as benchmark model the article "A new dataset of dog breed images and a benchmark for finegrained 
classification by Ding-Nan Zou1,2, Song-Hai Zhang1 (), Tai-Jiang Mu1, and Min Zhang". Where they achieved an 
accuracy of 82.65% of the model trained on Tsinghua Dogs achieved.


### Evaluation metrics
Validation Loss
Loss Function and Backpropagation 
Gradient Descent


### Project design
