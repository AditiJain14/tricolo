from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from tricolo.dataloader.dataset import ClrDataset, SnareTrainDataset, SnareValDataset

def backup_collate_fn(batch):
    data = default_collate([item for item in batch])
    model_id, category, arrays, pcd_array = data
    return model_id, category, arrays, pcd_array

def collate_fn(batch):
    data = default_collate([item for item in batch])
    model_id, voxels, category, arrays, images = data

    return model_id, voxels, category, arrays, images

def collate_fn_snare_train(batch):
    data = default_collate([item for item in batch])
    labels, arrays, images = data

    return labels, arrays, images.squeeze().float()

def collate_fn_snare_val(batch):
    data = default_collate([item for item in batch])
    entry_ids, arrays, images, images_2 = data

    return entry_ids, arrays, images, images_2

class DataSetWrapper(object):
    def __init__(self, dset, batch_size, stage, num_workers,  train_json_file, val_json_file, test_json_file, voxel_root_dir, image_size, voxel_size, transform):
        self.dset = dset
        self.batch_size = batch_size
        self.stage = 'train' if stage else 'test'
        self.num_workers = num_workers
        self.train_file = train_json_file
        self.val_file = val_json_file 
        self.test_file = test_json_file
        self.voxel_root_dir = voxel_root_dir
        self.image_size = image_size
        self.voxel_size = voxel_size
        self.transform = transform
        
    def get_data_loaders(self):
        train_dataset = ClrDataset(dset = self.dset, stage='train', json_file=self.train_file, voxel_root_dir=self.voxel_root_dir, image_size=self.image_size, voxel_size = self.voxel_size, transform=self.transform)
        valid_dataset = ClrDataset(dset = self.dset, stage='valid', json_file=self.val_file, voxel_root_dir=self.voxel_root_dir, image_size=self.image_size, voxel_size = self.voxel_size, transform=self.transform)
        test_dataset = ClrDataset(dset = self.dset, stage='test', json_file=self.test_file, voxel_root_dir=self.voxel_root_dir, image_size=self.image_size, voxel_size = self.voxel_size, transform=self.transform)

        train_loader, valid_loader, test_loader = self.get_train_val_test_data_loaders(train_dataset, valid_dataset, test_dataset)
        return train_loader, valid_loader, test_loader


    def get_train_val_test_data_loaders(self, train_dataset, valid_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=False, shuffle=True)
        return train_loader, valid_loader, test_loader


class DataSetWrapper_snare(object):
    def __init__(self, batch_size, stage, num_workers, train_json_file, val_json_file, test_json_file, voxel_root_dir, img_root_dir, image_size, voxel_size):
        self.batch_size = batch_size
        self.stage = 'train' if stage else 'test'
        self.num_workers = num_workers
        self.train_file = train_json_file
        self.val_file = val_json_file 
        self.test_file = test_json_file
        self.voxel_root_dir = voxel_root_dir
        self.img_root_dir = img_root_dir
        self.image_size = image_size
        self.voxel_size = voxel_size
    
    def get_data_loaders(self): 
        train_dataset = SnareTrainDataset(stage='train', batch_size=self.batch_size, json_file=self.train_file, voxel_root_dir=self.voxel_root_dir, img_root_dir=self.img_root_dir, image_size=self.image_size, voxel_size = self.voxel_size)
        valid_dataset = SnareValDataset(stage='valid', batch_size=self.batch_size, json_file=self.val_file, voxel_root_dir=self.voxel_root_dir, img_root_dir=self.img_root_dir, image_size=self.image_size, voxel_size = self.voxel_size)
        # TODO:
        test_dataset = SnareTrainDataset(stage='test', batch_size=self.batch_size, json_file=self.test_file, voxel_root_dir=self.voxel_root_dir, img_root_dir=self.img_root_dir, image_size=self.image_size, voxel_size = self.voxel_size)

        train_loader, valid_loader, test_loader = self.get_train_val_test_data_loaders(train_dataset, valid_dataset, test_dataset)
        return train_loader, valid_loader, test_loader

    def get_train_val_test_data_loaders(self, train_dataset, valid_dataset, test_dataset):
        train_loader = DataLoader(train_dataset, collate_fn=collate_fn_snare_train, batch_size=1, num_workers=self.num_workers, drop_last=True, shuffle=True)
        valid_loader = DataLoader(valid_dataset, collate_fn=collate_fn_snare_val, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True, shuffle=True)
        test_loader = DataLoader(test_dataset, collate_fn=collate_fn_snare_train, batch_size=1, num_workers=self.num_workers, drop_last=False, shuffle=True)
        return train_loader, valid_loader, test_loader