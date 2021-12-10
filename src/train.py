# from util import batch_accuracy, show_test_result
import torch
import torch.nn as nn
import numpy as np
from data_preprocessor import *
from UNet_model import *
from tqdm import tqdm
from datetime import datetime
from torchsummary import summary
import os
import copy
import torch.nn.functional as F
import sys
import matplotlib.pyplot as plt
DEVICE = 'cuda:0'
# device=torch.device('cuda:0')


def training(model, train_dataloader, val_dataloader, num_epochs, lr,
             weight_decay, momentum, batch_size):

    # model initialization
    number_of_class = 1
    model = model
    # summary(model, (3, 256, 256))
    criterion = nn.BCEWithLogitsLoss() if number_of_class == 1 else nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
   
    # Run your training / validation loops
    with tqdm(range(num_epochs), total=num_epochs, unit='Epoch') as pbar:
        best_validation_acc = 0
        for epoch in pbar:
            model.train()

            # training loop
            train_epoch_loss = 0
            valid_epoch_loss = 0
            #train_epoch_acc = 0
            #valid_epoch_acc = 0

            for batch_num, batch in enumerate(train_dataloader):
                train_batch_loss = 0
                img, label = batch
                # img, label = img.to(DEVICE), label.to(DEVICE)
                optimizer.zero_grad()
                prediction = model(img.float())
                if number_of_class != 1:
                    train_loss = criterion(prediction, label.type(torch.int64).squeeze(1))
                    # TODO calculate acc for instance segmentation
                else:
                    train_loss = criterion(prediction, label)
                    #train_acc = batch_accuracy(prediction, label)
                train_batch_loss += train_loss.item()
                train_epoch_loss += train_loss.item()
                #train_epoch_acc += train_acc
                train_loss.backward()
                optimizer.step()

            # validation loop
            with torch.no_grad():
                model.eval()
                valid_batch_loss = 0
                for batch_num, batch in enumerate(val_dataloader):
                    img, label = batch
                    # img, label = img.to(DEVICE), label.to(DEVICE)
                    validation_pred = model(img)
                    if number_of_class != 1:
                        val_loss = criterion(validation_pred, label.type(torch.int64).squeeze(1))
                        # TODO calculate acc for instance segmentation
                    else:
                        val_loss = criterion(validation_pred, label)
                        #val_acc = batch_accuracy(validation_pred, label)
                    valid_batch_loss += val_loss.item()
                    valid_epoch_loss += val_loss.item()
                    #valid_epoch_acc  += val_acc
            best_model_para = copy.deepcopy(model.state_dict())
            # Average accuracy
            #train_epoch_acc /= len(train_dataloader)
            #valid_epoch_acc /= len(val_dataloader)
            #if valid_epoch_acc > best_validation_acc:
                #best_validation_acc = valid_epoch_acc
                #best_model_para = copy.deepcopy(model.state_dict())
            #pbar.set_postfix(train_loss=train_epoch_loss, train_acc=train_epoch_acc, val_loss=valid_epoch_loss, val_acc=valid_epoch_acc)
            
        # save best epoch
        if not os.path.isdir("weights"):
            os.mkdir("weights")
        torch.save(best_model_para, os.path.join("weights", f'{datetime.strftime(datetime.now(), "%B-%d-%H")}.pt'))
        print(f'Best Validation Acc: {best_validation_acc}')

def prediction(test_dataloader,model, weights):
    nick_name = "tuge0"
    print(os.getcwd())
    weights_path = os.path.join("weights", weights)
    if not os.path.isfile(weights_path):
        raise Exception("weights file not found")
    model.cuda()
    model.eval()
    model.load_state_dict(torch.load(weights_path))
    result = []
    with tqdm(test_dataloader, unit='img') as pbar:
        for img in pbar:
            with torch.no_grad():
                img = img.to(DEVICE)
                image = img[10]
                image = image.cpu().numpy()
                plt.imshow(image.transpose(1,2,0))
                plt.show()
                pred = model(img)
                temp = pred.cpu().numpy()
                temp = temp[10]
                

                plt.imshow(temp.transpose(1,2,0))
                plt.show()
              
                if len(result) == 0:
                    result = torch.unsqueeze(pred, 0)
                    result = torch.squeeze(pred,1)
                else:
                    result = torch.cat((result, pred), 0)
                break
    print(f'prediction result: {result.shape}')
    sigmoid = torch.sigmoid(result)
    sigmoid[result > 0.5] = 1
    sigmoid[result <= 0.5] = 0
    torch.save(sigmoid, f'{nick_name}.pth')
