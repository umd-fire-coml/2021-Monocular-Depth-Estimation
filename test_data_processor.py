import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from data_preprocessor import Kitti
import pytest

@pytest.fixture

def my_data_generator():
    dataset = Kitti('images','depth_maps')
    return dataset
    
def test_len(my_data_generator):
    assert my_data_generator.__len__() == 5


