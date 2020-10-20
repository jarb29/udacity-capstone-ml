[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview

Dogs are closely involved in human lives as family members, and are very common as pets. On the other
hand, the number of dog-related incidents of injury and uncivilized behavior is increasing. This leads to a need for dog identification using modern visual technology, both for dog recognition and finer-grained
classification to breed.

Given an image of a dog, the algorithm will identify an estimate of the canine’s breed using Convolutional Neural Networks (CNN)!. If supplied an image of a human, the code will identify the resembling dog breed. In this project. Build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.

![Sample Output][image1]


## Project Statemet

The goal is create state-of-the-art CNN models for a dog classification. By piecing together a series of models designed to perform various tasks in a data processing pipeline.  

### Datasets and Inputs

2. https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip). The folder should contain 133 folders, each corresponding to a different dog breed.
3. http://vis-www.cs.umass.edu/lfw/lfw.tgz).  human dataset. 

### Solution statement

Steps to achieve this process:

	Step 0: Import Datasets
	Step 1: Detect Humans
	Step 2: Detect Dogs
	Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
	Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
	Step 5: Write your Algorithm
	Step 6: Test Your Algorithm

### A benchmark model
Stanford Dogs Dataset

Learning Attentive Pairwise Interaction for Fine-Grained Classification
24 Feb 2020 • Peiqin Zhuang • Yali Wang • Yu Qiao

Fine-grained classification is a challenging problem, due to subtle differences among highly-confused categories. Most approaches address this difficulty by learning discriminative representation of individual input image. On the other hand, humans can effectively identify contrastive clues by comparing image pairs. Inspired by this fact, this paper proposes a simple but effective Attentive Pairwise Interaction Network (API-Net), which can progressively recognize a pair of fine-grained images by interaction. Specifically, API-Net first learns a mutual feature vector to capture semantic differences in the input pair. It then compares this mutual vector with individual vectors to generate gates for each input image. These distinct gate vectors inherit mutual context on semantic differences, which allow API-Net to attentively capture contrastive clues by pairwise interaction between two images. Additionally, we train API-Net in an end-to-end manner with a score ranking regularization, which can further generalize API-Net by taking feature priorities into account. We conduct extensive experiments on five popular benchmarks in fine-grained classification. API-Net outperforms the recent SOTA methods, i.e., CUB-200-2011 (90.0%), Aircraft(93.9%), Stanford Cars (95.3%), Stanford Dogs (90.3%), and NABirds (88.1%).


### Evaluation metrics
Validation Loss
Loss Function and Backpropagation 
Gradient Descent


### Project design
