# Monocular Depth Estimation

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
## References
[TransDepth](https://github.com/syKevinPeng/TransDepth)

[UNet](https://github.com/syKevinPeng/UNet)
