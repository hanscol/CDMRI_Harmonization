import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import random
import os
import torch
import scipy.ndimage.filters as scfilter
from seg_map import *

class dataset(Dataset):
    def __init__(self, inputDir, segDir):
        self.seg = self.input_seg(inputDir[0], segDir)
        self.keys = [self.input(inputDir[0]), self.input(inputDir[1])]
        self.keys[0].sort()
        self.keys[1].sort()
        self.patch_size = [3,3,3,53]
        self.x = 0
        self.y = 0
        self.z = 0
        self.done = False
        self.next_image = True
        self.index = 0
        self.input_img = [np.zeros(1), np.zeros(1)]
        self.seg_map = seg_map23()

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

    def input_seg(self, inputDir, segDir):
        input_files = os.listdir(inputDir)
        seg_files = os.listdir(segDir)

        data_files = {}

        for i in input_files:
            for s in seg_files:
                subji = i.split('_')[0]
                subjs = s.split('_')[1]
                if subjs in subji:
                    data_files[os.path.join(inputDir,i)] = os.path.join(segDir,s)

        return data_files


    def get_size(self):
        return self.input_img[0].shape

    def simple_seg(self, seg_img):
        new_seg = np.zeros(seg_img.shape)
        for key in self.seg_map:
            new_seg[seg_img == key] = self.seg_map[key]
        return new_seg

    def smooth_seg(self):
        self.onehot = self.onehot.numpy()
        for i in range(self.onehot.shape[3]):
            self.onehot[:,:,:,i] = scfilter.gaussian_filter(self.onehot[:,:,:,i], sigma=2)
        self.onehot = torch.from_numpy(self.onehot)

    def image_step(self):
        if self.next_image:
            input_file = [self.keys[0][self.index], self.keys[1][self.index]]
            seg_file = self.seg[input_file[0]]

            self.input_img = [nib.load(input_file[0]).get_fdata(), nib.load(input_file[1]).get_fdata()]
            seg_img = nib.load(seg_file).get_fdata()
            seg_img = self.simple_seg(seg_img)
            seg_img = torch.from_numpy(seg_img).long()
            seg_img = seg_img.unsqueeze(3)
            self.onehot = torch.zeros([seg_img.shape[0], seg_img.shape[1], seg_img.shape[2], torch.max(seg_img) + 1])
            self.onehot = self.onehot.scatter(3, seg_img, 1)
            self.smooth_seg()

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
            patch[pymin:pymax, pxmin:pxmax, pzmin:pzmax, 30:] = self.onehot[ymin:ymax, xmin:xmax, zmin:zmax, :]

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
