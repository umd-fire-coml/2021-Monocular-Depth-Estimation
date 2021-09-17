# data preprocessor 

from torch.utils.data import DataLoader
import json

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
        