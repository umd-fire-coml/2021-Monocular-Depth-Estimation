# Monocular Depth Estimation

## Description
When an RGB image is inputted to the model, it produces a depth map that displays the predicted depth of each pixel. 
It is similar to that of a person's ability to percieve perspective, distinguishing what is far away and what is nearby.
It does this by evaluating the darkness of each pixel; something closer is generally lighter and something further is generally darker.

(Architecture description)

## Demo

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
