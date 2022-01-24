import jsonlines
import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset

class ClrDataset(Dataset):
    def __init__(self, dset, stage, json_file, voxel_root_dir, image_size, voxel_size, transform=None):
        self.dset = dset
        self.stage = stage
        self.clr_frame = []
        with jsonlines.open(json_file) as reader:
            self.clr_frame = list(reader)
        
        self.voxel_root_dir = voxel_root_dir
        self.transform = transform
        self.image_size = image_size
        self.voxel_size = voxel_size

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        model_id = self.clr_frame[idx]['model']
        category = self.clr_frame[idx]['category']

        path = '../data/all_npz/' + category + '/' + model_id + '.npz'
        data = np.load(path)

        if self.voxel_size == 32:
            voxel32 = data['voxel32']
            coords, colors = voxel32
            coords = coords.astype(int)

            voxels = np.zeros((4, 32, 32, 32))
            for i in range(coords.shape[0]):
                voxels[:3, coords[i, 0], coords[i, 1], coords[i, 2]] = colors[i]
                voxels[-1, coords[i, 0], coords[i, 1], coords[i, 2]] = 1
        elif self.voxel_size == 64:
            voxel64 = data['voxel64']
            coords, colors = voxel64
            coords = coords.astype(int)

            voxels = np.zeros((4, 64, 64, 64))
            for i in range(coords.shape[0]):
                voxels[:3, coords[i, 0], coords[i, 1], coords[i, 2]] = colors[i]
                voxels[-1, coords[i, 0], coords[i, 1], coords[i, 2]] = 1
        elif self.voxel_size == 128:
            voxel128 = data['voxel128']
            coords, colors = voxel128
            coords = coords.astype(int)

            voxels = np.zeros((4, 128, 128, 128))
            for i in range(coords.shape[0]):
                voxels[:3, coords[i, 0], coords[i, 1], coords[i, 2]] = colors[i]
                voxels[-1, coords[i, 0], coords[i, 1], coords[i, 2]] = 1
        else:
            raise('Not supported voxel size')

        images = data['images']
        if self.image_size != 224:
            resized = []
            for i in range(images.shape[0]):
                image = images[i].transpose(1, 2, 0)
                image = cv.resize(image, dsize=(self.image_size, self.image_size))
                resized.append(image)
            resized = np.array(resized)
            images = resized.transpose(0, 3, 1, 2)
        
        text = self.clr_frame[idx]['caption']
        text = text.replace("\n", "")

        arrays = np.asarray(self.clr_frame[idx]['arrays'])

        return model_id, voxels.astype(np.float32), category, arrays, images.astype(np.float32)


# train: 59777, val: 7435, test:7452
# chair: 32776, table: 41888
# total: 74664



