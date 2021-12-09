# Monocular Depth Estimation

## Description

## Demo

## Colab notebook

## Directory Guide
1. .github/workflows

2. dataset

3. src: directory containing our code

UNet_model.py

data_preprocessor.py

environment.yml: a yaml file that lists required packages to install to set up the environment

model_summary.py

train.py: trains and tests the model
  
4. test: directory containing test cases

5. .gitignore

6. Kitti.json

7. requirements.txt: lists out all the installed packages and version numbers

8. test-requirements.txt: lists out required packages to run tests


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

## How to get the training started and get the trained weights

## How to test the model and get the predicted results

## Citations

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
