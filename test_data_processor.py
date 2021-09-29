import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from data_preprocessor import Kitti
import pytest

@pytest.fixture

def my_data_generator():
    return datasets.Kitti
    
def test_imports():
    assert True == True


