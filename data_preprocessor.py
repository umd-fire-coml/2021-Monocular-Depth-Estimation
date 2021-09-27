# data preprocessor 

from torch.utils.data import DataLoader

import json
import os
import pandas as pd
from torchvision.io import read_image

class Kitti(Dataset):
    def __init__(self, annotations_file, img_dir, transform = None, target_transform = None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_lables.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
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
        features, lables = i
        img = features[0].squeeze()
        label = label[0]
        toReturn[img] = label
    return toReturn
        
