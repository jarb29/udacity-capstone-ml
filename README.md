[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


## Project Overview
   Given an image of a dog, the algorithm will identify an estimate of the canine’s breed using Convolutional Neural Networks (CNN)!. If supplied an image of a human, the code will identify the resembling dog breed. In this project. Build a pipeline that can be used within a web or mobile app to process real-world, user-supplied images.

![Sample Output][image1]


## Project Statemet

The goal is create state-of-the-art CNN models for a dog classification. By piecing together a series of models designed to perform various tasks in a data processing pipeline.  

Steps to achieve this process:

	Step 0: Import Datasets
	Step 1: Detect Humans
	Step 2: Detect Dogs
	Step 3: Create a CNN to Classify Dog Breeds (from Scratch)
	Step 4: Create a CNN to Classify Dog Breeds (using Transfer Learning)
	Step 5: Write your Algorithm
	Step 6: Test Your Algorithm

### Datasets and Inputs

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`.  The `dogImages/` folder should contain 133 folders, each corresponding to a different dog breed.
3. Download the [human dataset](http://vis-www.cs.umass.edu/lfw/lfw.tgz).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 



## Project Submission

Your submission should consist of the github link to your repository.  Your repository should contain:
- The `dog_app.ipynb` file with fully functional code, all code cells executed and displaying output, and all questions answered.
- An HTML or PDF export of the project notebook with the name `report.html` or `report.pdf`.

Please do __NOT__ include any of the project data sets provided in the `dogImages/` or `lfw/` folders.


