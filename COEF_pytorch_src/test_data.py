import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import random
import os
import torch
import scipy.ndimage.filters as scfilter


class dataset(Dataset):
    def __init__(self, inputDir):
        self.keys = [self.input(inputDir[0]), self.input(inputDir[1])]
        self.keys[0].sort()
        self.keys[1].sort()
        self.patch_size = [3,3,3,30]
        self.x = 0
        self.y = 0
        self.z = 0
        self.done = False
        self.next_image = True
        self.index = 0
        self.input_img = [np.zeros(1), np.zeros(1)]


        self.x_offset = int(self.patch_size[1] / 2)
        self.y_offset = int(self.patch_size[0] / 2)
        self.z_offset = int(self.patch_size[2] / 2)

    def __len__(self):
        return 100000000

    def input(self, inputDir):
        input_files = os.listdir(inputDir)

        data_files = []

        for i in input_files:
            data_files.append(os.path.join(inputDir,i))

        return data_files


    def get_size(self):
        return self.input_img[0].shape


    def image_step(self):
        if self.next_image:
            input_file = [self.keys[0][self.index], self.keys[1][self.index]]

            self.input_img = [nib.load(input_file[0]).get_fdata(), nib.load(input_file[1]).get_fdata()]


            self.next_image = False

        else:
            self.z += 1

            if self.z >= self.input_img[0].shape[2]:
                self.z = 0
                self.y += 1
            if self.y >= self.input_img[0].shape[0]:
                self.y = 0
                self.x += 1
            if self.x >= self.input_img[0].shape[1]:
                self.x = 0
                self.next_image = True
                self.index += 1
                if self.index >= len(self.keys[0]):
                    self.done = True
                    self.index -= 1
                else:
                    self.image_step()

    def get_patch(self):
        self.image_step()
        while np.sum(self.input_img[0][self.y, self.x, self.z, :] + self.input_img[1][self.y, self.x, self.z,:]) == 0 and not self.done:
            self.image_step()

        if not self.done:
            patch = np.zeros(self.patch_size)

            xmin = max(self.x - self.x_offset, 0)
            xmax = min(self.x + self.x_offset, self.input_img[0].shape[1] - 1)
            ymin = max(self.y - self.y_offset, 0)
            ymax = min(self.y + self.y_offset, self.input_img[0].shape[0] - 1)
            zmin = max(self.z - self.z_offset, 0)
            zmax = min(self.z + self.z_offset, self.input_img[0].shape[2] - 1)

            pxmin = self.x_offset - (self.x - xmin)
            pxmax = self.x_offset + (xmax - self.x)
            pymin = self.y_offset - (self.y - ymin)
            pymax = self.y_offset + (ymax - self.y)
            pzmin = self.z_offset - (self.z - zmin)
            pzmax = self.z_offset + (zmax - self.z)

            patch[pymin:pymax, pxmin:pxmax, pzmin:pzmax, 0:15] = self.input_img[0][ymin:ymax, xmin:xmax, zmin:zmax, :]
            patch[pymin:pymax, pxmin:pxmax, pzmin:pzmax, 15:30] = self.input_img[1][ymin:ymax, xmin:xmax, zmin:zmax, :]

            return patch
        else:
            return None


    def __getitem__(self, idx):

        input = self.get_patch()

        if not self.done:
            input = input.transpose((3,0,2,1))

            input = input.astype(np.float32)

            input[np.isnan(input)] = 0
            input[np.isinf(input)] = 0

        return {'input': input, 'xyz': [self.x, self.y, self.z], 'input_img': [self.keys[0][self.index], self.keys[1][self.index]]}
