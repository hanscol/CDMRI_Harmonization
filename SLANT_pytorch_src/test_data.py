import numpy as np
from torch.utils.data import Dataset
import nibabel as nib
import random
import os
import torch

class dataset(Dataset):
    def __init__(self, inputDir, segDir):
        self.seg = self.input_seg(inputDir[0], segDir)
        self.keys = [self.input(inputDir[0]), self.input(inputDir[1])]
        self.keys[0].sort()
        self.keys[1].sort()
        self.patch_size = [3,3,3,238]
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

    # def randomCrop(self, input, target):
    #     x = random.randint(0, input.shape[1] - self.patch_size[0])
    #     y = random.randint(0, input.shape[0] - self.patch_size[1])
    #     z = random.randint(0, input.shape[2] - self.patch_size[2])
    #     input = input[y:y + self.patch_size[1], x:x + self.patch_size[0], z:z + self.patch_size[2], :]
    #     target = target[y:y + self.patch_size[1], x:x + self.patch_size[0], z:z + self.patch_size[2], :]
    #     target = target[int(self.patch_size[1]/2)+1, int(self.patch_size[0]/2)+1, int(self.patch_size[2]/2)+1, :]
    #     return input, target

    def get_size(self):
        return self.input_img[0].shape

    def image_step(self):
        if self.next_image:
            input_file = [self.keys[0][self.index], self.keys[1][self.index]]
            seg_file = self.seg[input_file[0]]

            self.input_img = [nib.load(input_file[0]).get_fdata(), nib.load(input_file[1]).get_fdata()]
            seg_img = nib.load(seg_file).get_fdata()
            seg_img = torch.from_numpy(seg_img).long()
            seg_img = seg_img.unsqueeze(3)
            self.onehot = torch.zeros([seg_img.shape[0], seg_img.shape[1], seg_img.shape[2], 208])
            self.onehot = self.onehot.scatter(3, seg_img, 1)


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

            # for x in range(self.x - x_offset, self.x + x_offset + 1):
            #     for y in range(self.y - y_offset, self.y + y_offset + 1):
            #         for z in range(self.z - z_offset, self.z + z_offset + 1):
            # for x in range(self.patch_size[1]):
            #     for y in range(self.patch_size[0]):
            #         for z in range(self.patch_size[2]):
            #             imgx = self.x - self.x_offset + x
            #             imgy = self.y - self.y_offset + y
            #             imgz = self.z - self.z_offset + z
            #
            #             xcheck = imgx < self.input_img[0].shape[1] and imgx >= 0
            #             ycheck = imgy < self.input_img[0].shape[0] and imgy >= 0
            #             zcheck = imgz < self.input_img[0].shape[2] and imgz >= 0
            #             if xcheck and ycheck and zcheck:
            #                 patch[y, x, z, 0:15] = self.input_img[0][imgy, imgx, imgz, 0:15]
            #                 patch[y, x, z, 15:30] = self.input_img[1][imgy, imgx, imgz, 0:15]
            #                 patch[y, x, z, 30:] = self.onehot[:, imgy, imgx, imgz]

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
