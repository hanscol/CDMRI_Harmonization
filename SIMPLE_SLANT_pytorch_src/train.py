from __future__ import print_function, division
from batch import *
from data import *
from models import *
from resnet3d import *
import sys

import torch
from torch.utils.data import Dataset
from torchvision import transforms



def main():
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    inputDir = [sys.argv[1] + '/1200/source', sys.argv[1] + '/3000/source']
    targetDir = [sys.argv[1] + '/1200/target', sys.argv[1] + '/3000/target']
    segDir = sys.argv[1] + '/SLANT'

    train_loss_file = sys.argv[2] + '/train_loss.txt'

    if int(sys.argv[3]) == -1:
        f = open(train_loss_file, 'w')
        f.close()

    train_dataset = dataset(inputDir, targetDir, segDir)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10000, shuffle=False, **kwargs)

    model_file = sys.argv[2] + '/epoch_{}'
    #model = UNet().to(device)
    model = resnet10(sample_size=1, sample_duration=1).to(device)

    start_epoch = 1
    if int(sys.argv[3]) != -1:
        model.load_state_dict(torch.load(model_file.format(sys.argv[3])))
        start_epoch = int(sys.argv[3]) + 1

    lr = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)


    for epoch in range(start_epoch, 31):
        print('\nEpoch {}: '.format(epoch))

        train_loss = train(model, device, train_loader, optimizer)

        with open(train_loss_file, "a") as file:
            file.write(str(train_loss))
            file.write('\n')

        if epoch%3==0:
            with open(model_file.format(epoch), 'wb') as f:
                torch.save(model.state_dict(), f)



if __name__ == '__main__':
    main()
