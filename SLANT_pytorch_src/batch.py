from __future__ import print_function, division
import os
import torch

import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import nibabel as nib
import warnings
warnings.filterwarnings("ignore")

def train(model, device, train_loader, optimizer):
    model.train()

    train_loss = 0
    for batch_idx, sample in enumerate(train_loader):
        data = sample['input']
        target = sample['target']

        data = data.to(device)
        optimizer.zero_grad()

        output = model(data)

        target= target.to(device)
        loss_fun = torch.nn.MSELoss()
        #loss = loss_fun(output[:,0], target[:,0]) +\
        #       10*loss_fun(output[:,1:6], target[:,1:6]) +\
        #       50*loss_fun(output[:,6:], target[:,6:])
        loss = loss_fun(output, target)


        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset) / train_loader.batch_size
    print('\tTraining set: Average loss: {:.4f}'.format(train_loss), end='')
    return train_loss
