from __future__ import print_function, division
from batch import *
from test_data import *
from models import *
from resnet3d import *
import sys
import numpy as np
import nibabel as nib
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms


def main():
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(1)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    inputDir = [sys.argv[1]+'1200_test', sys.argv[1]+'3000_test']
    segDir = sys.argv[1] + 'SLANT_test'


    test_dataset = dataset(inputDir, segDir)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=5, shuffle=True, **kwargs)

    model_file = sys.argv[2] + '/epoch_{}'
    # model = UNet().to(device)
    model = resnet10(sample_size=1, sample_duration=1).to(device)

    model.load_state_dict(torch.load(model_file.format(sys.argv[3])))

    model.eval()

    i=0
    batch_size = 100
    out_img = np.zeros(1)
    prev_input = ''
    with torch.no_grad():
        while not test_dataset.done:
            init = True
            xyz = []
            input_imgs = []



            data_left = False
            for j in range(batch_size):
                sample = test_dataset[j]

                if not test_dataset.done:
                    data_left = True
                    if init:
                        data = torch.from_numpy(sample['input']).unsqueeze(0)
                        if i == 0:
                            prev_input = sample['input_img']
                        init = False
                    else:
                        data = torch.cat((data, torch.from_numpy(sample['input']).unsqueeze(0)), 0)
                    xyz.append(sample['xyz'])
                    input_imgs.append(sample['input_img'])
                else:
                    break

            if data_left:
                data = data.to(device)
                output = model(data).cpu()
                output = output.numpy()

                if i == 0:
                    out_img1 = np.zeros(test_dataset.get_size())
                    out_img2 = np.zeros(test_dataset.get_size())

                for j in range(output.shape[0]):

                    if input_imgs[j] != prev_input or j == output.shape[0]-1 and test_dataset.done:
                        nib_img = nib.Nifti1Image(out_img1, nib.load(prev_input[0]).affine, nib.load(prev_input[0]).header)
                        nib.save(nib_img, os.path.join(sys.argv[2], prev_input[0].split('/')[-1]))
                        nib_img = nib.Nifti1Image(out_img2, nib.load(prev_input[1]).affine, nib.load(prev_input[1]).header)
                        nib.save(nib_img, os.path.join(sys.argv[2], prev_input[1].split('/')[-1]))
                        prev_input = input_imgs[j]
                        out_img1 = np.zeros(test_dataset.get_size())
                        out_img2 = np.zeros(test_dataset.get_size())

                    out_img1[xyz[j][1], xyz[j][0], xyz[j][2], :] = output[j,0:15]
                    out_img2[xyz[j][1], xyz[j][0], xyz[j][2], :] = output[j,15:30]

                i += 1


if __name__ == '__main__':
    main()
