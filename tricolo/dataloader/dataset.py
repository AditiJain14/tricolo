import jsonlines
import cv2 as cv
import numpy as np
import json
import os
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

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


class SnareTrainDataset(Dataset): 
    def __init__(self, stage, batch_size, json_file, voxel_root_dir, img_root_dir, image_size, voxel_size): 
        self.stage = stage
        self.train_json = json_file
        self.img_folder = img_root_dir
        with open(self.train_json) as f:
            self.data = json.load(f)
        self.image_size = image_size
        self.bs = batch_size

    def __len__(self):
        return len(self.data)
    
    def get_images(self, modelid):
        view_ids = [f"{modelid}-{i}.png" for i in range(6,14)]
        pre_path = os.path.join(self.img_folder, modelid)
        img_paths = [os.path.join(pre_path,view_id) for view_id in view_ids]
        images= []
        for img_path in img_paths:
            img = cv.imread(img_path)
            resize_img = cv.resize(img, (self.image_size, self.image_size))   
            resize_img = np.array(resize_img).astype(np.float32) / 255.
            images.append(resize_img)
        images = np.array(images).transpose(0, 3, 1, 2) # 8， 3， 128， 128
        
        return images
    
    # def __getitem__(self, idx):
    # Use paired images
    #     entry = self.data[idx]
        
    #     # get text
    #     anno = np.array(entry['array'])
        
    #     # Train. (Test doesn't have answers)
    #     entry_idx = entry['ans'] 
        
    #     if len(entry['objects']) == 2:
    #         key1, key2 = entry['objects']
    #     # fix missing key in pair
    #     else:
    #         key1 = entry['objects'][entry_idx]
    #         while True:
    #             new_idx = np.random.choice(range(len(self.data)))
    #             key2 = self.data[new_idx]['objects'][0]
    #             if key2 != key1:
    #                 break
        
    #     # process TARGET image. original 512 x 512. convert to 224 x 224
    #     images_1 = self.get_images(str(key1)) # 8,3, 128, 128
    #     images_2 = self.get_images(str(key2))
        
    #     all_images = np.stack((images_1, images_2)) # 2, 8, 3, 128, 128
        
    #     if entry_idx == 0:
    #         label = np.array([1, 0], dtype=int)
    #     else:
    #         label = np.array([0, 1], dtype=int)

    #     return label, anno, all_images
    
    def __getitem__(self, idx):
        # use paired images with bs-2 other images
        entry = self.data[idx]
        
        # get text
        anno = np.array(entry['array'])
        
        # Train. (Test doesn't have answers)
        entry_idx = entry['ans']
        modelid = entry['objects'][entry_idx]
        
        # process TARGET image. original 512 x 512. convert to 224 x 224
        images = self.get_images(modelid)
        
        ### Find another self.bs-1 different models
        candidates = []
        if len(entry['objects'])>1:
            modelid2 = entry['objects'][1-entry_idx]
            candidates.append(modelid2)
        
        while len(candidates)<self.bs-1:
            new_idx = np.random.choice(range(len(self.data)))
            another_entry = self.data[new_idx]
            modelid2 = another_entry["objects"][0]
            if modelid2 != modelid:
                candidates.append(modelid2)
        
        pos = np.random.randint(0, self.bs)
        label = np.zeros(self.bs, dtype=int)
        label[pos] = 1
        all_images = np.zeros((self.bs, 8, 3, self.image_size, self.image_size))
        all_images[pos] = images
        
        for i in range(self.bs-1):
            if i==pos:
                continue 
            modelid2 = candidates[i]
            images_2 = self.get_images(modelid2)
            all_images[i] = images_2
        
        return label, anno, all_images

class SnareValDataset(Dataset): 
    def __init__(self, stage, batch_size, json_file, voxel_root_dir, img_root_dir, image_size, voxel_size): 
        self.stage = stage
        self.train_json = json_file
        self.img_folder = img_root_dir
        with open(self.train_json) as f:
            self.data = json.load(f)
        self.image_size = image_size
        self.bs = batch_size

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        entry = self.data[idx]
        
        # get text
        anno = np.array(entry['array'])
        
        # Train. (Test doesn't have answers)
        entry_idx = entry['ans']
        modelid = entry['objects'][entry_idx]
        
        # process TARGET image
        # original 512 x 512
        # convert to 224 x 224
        view_ids = [f"{modelid}-{i}.png" for i in range(6,14)]
        pre_path = os.path.join(self.img_folder, modelid)
        img_paths = [os.path.join(pre_path,view_id) for view_id in view_ids]
        images= []
        for img_path in img_paths:
            img = cv.imread(img_path)
            resize_img = cv.resize(img, (self.image_size, self.image_size))
            resize_img = np.array(resize_img).astype(np.float32) / 255.
            images.append(resize_img)
        images = np.array(images).transpose(0, 3, 1, 2) # 8， 3， 128， 128
        
        ### Find anotherdifferent models
        if len(entry['objects'])>1:
            modelid2 = entry['objects'][1-entry_idx]
        
        while True:
            new_idx = np.random.choice(range(len(self.data)))
            another_entry = self.data[new_idx]
            modelid2 = another_entry["objects"][0]
            if modelid2 != modelid:
                break

        view_ids = [f"{modelid2}-{i}.png" for i in range(6,14)]
        pre_path = os.path.join(self.img_folder, modelid2)
        img_paths = [os.path.join(pre_path,view_id) for view_id in view_ids]
        images_2= []
        for img_path in img_paths:
            img = cv.imread(img_path)
            resize_img = cv.resize(img, (self.image_size, self.image_size))
            resize_img = np.array(resize_img).astype(np.float32) / 255.
            images_2.append(resize_img)
        images_2 = np.array(images_2).transpose(0, 3, 1, 2) 

        
        return entry_idx, anno, images, images_2


if __name__ == "__main__":
    # dset = SnareTrainDataset()
    # loader = DataLoader(dset, batch_size=16, num_workers=4, drop_last=True, shuffle=True)
    # for anno, image in loader:
    #     print(anno, image.shape)
    img_folder = "../data/snare/shapenet-sem/screenshots"
    json_file = "../data/snare/amt/processed_json/processed_train.json"
    with open(json_file, 'r') as f: 
        data = json.load(f)
    for entry in tqdm(data):
        modelid = entry["objects"][1]
        modelid = "c28ae120a2d2829e50e9662c8e47fff"
        view_ids = [f"{modelid}-{i}.png" for i in range(6,14)]
        pre_path = os.path.join(img_folder, modelid)
        img_paths = [os.path.join(pre_path,view_id) for view_id in view_ids]
        images= []
        for img_path in img_paths:
            img = cv.imread(img_path)
            try:
                resize_img = cv.resize(img, (128, 128))   
            except Exception as e: 
                print(modelid)
                