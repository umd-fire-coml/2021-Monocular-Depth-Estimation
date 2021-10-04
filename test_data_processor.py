import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from data_preprocessor import Kitti
import pytest

@pytest.fixture


def my_data_generator():
    dataset = Kitti('images','depth_maps')
    return dataset
    
def test_len(my_data_generator):
    assert my_data_generator.__len__() == 5

def test_get_item(my_data_generator):
    my_data_generator.__getitem__(0)
    assert True == True

def test_display(my_data_generator):
    img, label = my_data_generator.__getitem__(0)
    plt.imshow(img)
    plt.show()
    plt.imshow(label)
    plt.show()
    

    assert True == True




