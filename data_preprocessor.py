# data preprocessor 

from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import json
import os
import pandas as pd
from torchvision.io import read_image

class Kitti(Dataset):
    def __init__(self, img_dir, depth_maps, transform = None, target_transform = None):
        self.depth_maps = depth_maps
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.depth_maps)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.depth_maps.iloc[idx, 0])
        image = read_image(img_path)
        label = read_image(self.depth_maps)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
       

def process_data(training_data, testing_data, train_file, test_file):
    train_dataloader = DataLoader(training_data, batch_size = 64, shuffle = True)
    test_dataloader = DataLoader(testing_data, batch_size = 64, shuffle = True)

    train_dict = get_dict(train_dataloader)
    test_dict = get_dict(test_dataloader)

    with open(train_file, "w") as outfile:
        json.dump(train_dict, outfile)
    
    with open(test_file, "w") as outfile:
        json.dump(test_dict, outfile)

def get_dict(dataloader):
    toReturn = {}
    for i in dataloader:
        features, labels = i
        img = features[0].squeeze()
        label = labels[0]
        toReturn[img] = label
    return toReturn
        
