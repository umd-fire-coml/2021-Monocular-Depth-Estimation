# Monocular Depth Estimation

## Description
When an RGB image is inputted to the model, it produces a depth map that displays the predicted depth of each pixel. 
It is similar to that of a person's ability to percieve perspective, distinguishing what is far away and what is nearby.
It does this by evaluating the darkness of each pixel; something closer is generally lighter and something further is generally darker.

For the model architecture, we chose to use a UNet model. It was first proposed for abnormality localization in medical images that used convolutional networks for Biomedical Image Segmentation. As Manikandan [2021] explained, it has the ability for pixel-level localization and distinguishes unique patterns. It also has a 'U' shaped structure with the first half being an encoder and the last half being a decoder. Purkayastha [2020] also described that, "[t]his architecture consists of three sections: The contraction, The bottleneck, and the expansion section". 

## Demo

In this demo, our predicted results is a black picture. This is due to the time constraint we had to train. 
Because we did not have enough time to train our model, the results are as they are; however, with more time, our results would have been better.


https://user-images.githubusercontent.com/60985308/145638324-f7cc404d-e327-4e24-b335-5d89d3bd447a.mp4


## Colab notebook
[Google Colab Notebook](https://colab.research.google.com/drive/13TtdaET7ODnF2ewG49d50OzSDFau9qWG?usp=sharing)

## Directory Guide
1. .github/workflows/run_all_tests.yml: runs all test cases in test directory

2. dataset: directory containing the dataset

dataset/depth_maps: directory containing corresponding annotated depth maps to images in images directory (used while training)

dataset/images: directory containing raw images

3. src: directory containing our code

src/UNet_model.py: defines and builds the UNet model

src/data_preprocessor.py: preprocesses data from Kitti dataset

src/environment.yml: a yaml file that lists necessary packages to set up the environment

src/model_summary.py: generates summary of the model

src/train.py: trains and tests the model
  
4. test: directory containing test cases for the code in src directory

5. requirements.txt: lists out all the installed packages and version numbers

6. test-requirements.txt: lists out required packages to run tests


## How to install the environment
1. Make sure you are in the 2021-Monocular-Depth-Estimation directory
```bash
cd 2021-Monocular-Depth-Estimation
```
2. To install the environment, run this command:
```bash
conda env create -f src/environment.yml
```
3. Activate the new environment with this command:
```bash
conda activate monocular-depth-estimation
```
4. If you need to deactivate the environment, run this command:
```bash
conda deactivate
```

## How to download or check the dataset
To download raw data: Click [here](http://www.cvlibs.net/datasets/kitti/raw_data.php) and download the raw data download script and run the .sh file

To download depth maps: Click [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and download the annotated depth maps dataset

## How to get the training started and get the trained weights
1. Open train.py

2. Go to main and set mode variable to "train"

3. Run train.py

4. Trained weights will be saved in a folder named "weights"

## How to test the model and get the predicted results
1. Open train.py

2. Go to main and set mode variable to "test"

3. Run train.py

4. Predicted results will be saved in a .ph file

## Citations
[1] Monimoy Purkayastha. 2020. Monocular Depth Estimation and Background/Foreground Extraction using UNet Deep Learning Architecture. (July 2020). Retrieved December 10, 2021 from https://medium.com/analytics-vidhya/monocular-depth-estimation-and-background-foreground-extraction-using-unet-deep-learning-bdfd19909aca 

[2] Bala Manikandan. 2021. Monocular depth estimation using U-Net. (July 2021). Retrieved December 10, 2021 from https://medium.com/mlearning-ai/monocular-depth-estimation-using-u-net-6f149fc34077 

Code References:

[TransDepth](https://github.com/syKevinPeng/TransDepth)

[UNet](https://github.com/syKevinPeng/UNet)

[nikhilroxtomar](https://github.com/nikhilroxtomar/Semantic-Segmentation-Architecture/blob/main/PyTorch/unet.py)
