
import sys
import os
import matplotlib.pyplot as plt
sys.path.append('src')


from data_preprocessor import Kitti
import pytest


@pytest.fixture

def my_data_generator():
    dataset = Kitti('dataset/images','dataset/depth_maps')
    return dataset

def test_len(my_data_generator):
    assert my_data_generator.__len__() == 6

def test_tensors(my_data_generator):
    img, label = my_data_generator[0]
    print(img.float())
    print(label)
    assert True

def test_get_item(my_data_generator):
    my_data_generator[0]
    assert True

# def test_display(my_data_generator):
#     img, label = my_data_generator[0]
#     print(len(img))
#     print(len(label))
#     plt.imshow(img)
#     plt.show()
#     plt.imshow(label)
#     plt.show()
#     assert True





    





