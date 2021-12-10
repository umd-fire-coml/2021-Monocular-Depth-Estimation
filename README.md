# Monocular Depth Estimation

## Description
When an RGB image is inputted to the model, it produces a depth map that displays the predicted depth of each pixel. 
It is similar to that of a person's ability to percieve perspective, distinguishing what is far away and what is nearby. 

## Demo

## Colab notebook
https://colab.research.google.com/drive/13TtdaET7ODnF2ewG49d50OzSDFau9qWG?usp=sharing

## Directory Guide
1. .github/workflows/run_all_tests.yml: runs all test cases in test directory

2. dataset: directory containing the training dataset

dataset/depth_maps: directory containing corresponding depth maps to images in images directory

dataset/images: directory containing images

3. src: directory containing our code

src/UNet_model.py: defines and builds the UNet model

src/data_preprocessor.py: preprocesses data from Kitti dataset

src/environment.yml: a yaml file that lists required packages to install to set up the environment

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
```bash
To download depth maps: Click [here](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) and download the annotated depth maps dataset
```
## How to get the training started and get the trained weights
1. Open train.py
```bash
2. Go to main and set mode variable to "train"
```bash
3. Run train.py
```bash
4. Trained weights will be saved in a folder named "weights"
```
## How to test the model and get the predicted results
1. Open train.py
```bash
2. Go to main and set mode variable to "test"
```bash
3. Run train.py
```bash
4. predicted results will be saved in a .ph file
```
## Citations


# USE THIS FOR REFERENCE - DELETE LATER:
## How to use Conda
1. Create Conda envrionment
```bash
conda create --name your_env_name

conda create -nname your_env_name "python==3.8" # you can specify the python version when create your envrionment
```
2. Activate Conda envrionment you just created
```bash
conda activate your_env_name  # you have to activate the envrionment before using it
```
3. Install packeges within the envrionment
```bash
conda install numpy matplotlib
```
4. Install pytorch according the pytorch [website](https://pytorch.org/get-started/locally/)
5. Running python under the env. Specify the python version if you needed
```bash
python3.8 your_script.py
```
6. Deactivate your conda env
```bash
conda deactivate
```
7. check all your install packages
```bash
conda list

conda list > requirements.txt # export all your installed packages to a file
```
8. check all your conda env
``` bash
conda env list
```
