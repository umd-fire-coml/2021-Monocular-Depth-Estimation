# data preprocessor 
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
import numpy
import os
import random
import sys


class Kitti(data.Dataset):
    def __init__(self, img_dir, depth_maps, transforms=True):
        self.depth_maps = depth_maps
        self.img_dir = img_dir
        self.transforms = transforms
        process_data(depth_maps, img_dir, 'Kitti.json')

    def __len__(self):
        

        return len(json.load(open('Kitti.json')))
    
    def __getitem__(self, idx):

        data = json.load(open('Kitti.json'))


        img_path = data[str(idx)][1]
        depth_dir = data[str(idx)][0]
      
        image = Image.open(img_path)
        label = Image.open(depth_dir)

        image_width, image_height = image.size
        

        x = random.randint(0, image_width - 256)
        y = random.randint(0, image_height - 256)

        image = image.crop((x, y, x + 256, y + 256))
        label = label.crop((x, y, x + 256, y + 256))

        if self.transforms:
            
            label = label.convert('RGB')
            image = image.convert('RGB')
            


            transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
                                ),
                        ])
            image = transform(image)
            label = transform(label)

            
        return image, label

def preprocessing(batch_size, is_img_aug=True):

    sys.path.append('dataset')
    train_set = Kitti('dataset/images', 'dataset/depth_maps')
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_set = Kitti('dataset/images', 'dataset/depth_maps')
    val_dataloader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_set = Kitti('dataset/images', 'dataset/depth_maps')
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    all_set = Kitti('dataset/images', 'dataset/depth_maps')
    all_dataloader = torch.utils.data.DataLoader(all_set, batch_size=batch_size, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, all_dataloader







def get_file_number(filename):
    toReturn = ""
    foundStart = False
    for element in filename:
        if not foundStart:
            if element != "0":
                toReturn += element
                foundStart = True
        else:   
            toReturn += element
    return toReturn
            
    
def add_zeroes(filename, length):
    while len(filename) < length:
        filename = '0' + filename
    return filename

def get_img_file(filename):
    return add_zeroes(get_file_number(filename), 10)

    

def process_data(label_path, img_path, file):
    dict = {}
    count = 0
    for filename in os.listdir(label_path):
        if (filename[len(filename) - 3:] == "png"):
            dict[count] = (os.path.join(label_path, filename), os.path.join(img_path, get_img_file(filename)))
            count += 1
    
    with open(file, "w") as outfile:
        json.dump(dict, outfile)
    


    
        
