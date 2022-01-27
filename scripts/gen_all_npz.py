import os
import nrrd
import jsonlines
import cv2 as cv
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

train_json_file =   '../data/text2shape-data/shapenet/train_map.jsonl'
val_json_file =     '../data/text2shape-data/shapenet/val_map.jsonl'
test_json_file =    '../data/text2shape-data/shapenet/test_map.jsonl'
voxel32_root_dir =  '../data/shapenet/nrrd_256_filter_div_32_solid'
voxel64_root_dir =  '../data/shapenet/nrrd_256_filter_div_64_solid'
voxel128_root_dir = '../data/shapenet/nrrd_256_filter_div_128_solid'

train_dataset = []
with jsonlines.open(train_json_file) as reader:
    train_dataset = list(reader)

val_dataset = []
with jsonlines.open(val_json_file) as reader:
    val_dataset = list(reader)

test_dataset = []
with jsonlines.open(test_json_file) as reader:
    test_dataset = list(reader)

print('Train has {} samples.'.format(len(train_dataset)))
print('Val has {} samples.'.format(len(val_dataset)))
print('Test has {} samples.'.format(len(test_dataset)))

def save_npz(obj):
    category = obj['category']
    model_id = obj['model']

    save_dir = '../data/all_npz' + '/' + category
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ######################################################################################################
    '''
    voxel32
    '''
    voxel32, _ = nrrd.read(voxel32_root_dir + '/' + model_id + '/' + model_id + '.nrrd')
    voxel32 = voxel32.astype(np.float32)
    voxel32 = voxel32 / 255.

    mask = (voxel32[-1] > 0.9).astype(int)
    index = np.nonzero(mask)

    x = index[0].reshape(-1, 1)
    y = index[1].reshape(-1, 1)
    z = index[2].reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    r = voxel32[0][index].reshape(-1, 1)
    g = voxel32[1][index].reshape(-1, 1)
    b = voxel32[2][index].reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)
    voxel32 = [coords, colors]
    ######################################################################################################

    ######################################################################################################
    '''
    voxel64
    '''
    voxel64, _ = nrrd.read(voxel64_root_dir + '/' + model_id + '/' + model_id + '.nrrd')
    voxel64 = voxel64.astype(np.float32)
    voxel64 = voxel64 / 255.

    mask = (voxel64[-1] > 0.9).astype(int)
    index = np.nonzero(mask)

    x = index[0].reshape(-1, 1)
    y = index[1].reshape(-1, 1)
    z = index[2].reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    r = voxel64[0][index].reshape(-1, 1)
    g = voxel64[1][index].reshape(-1, 1)
    b = voxel64[2][index].reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)
    voxel64 = [coords, colors]
    ######################################################################################################

    ######################################################################################################
    '''
    voxel128
    '''
    voxel128, _ = nrrd.read(voxel128_root_dir + '/' + model_id + '/' + model_id + '.nrrd')
    voxel128 = voxel128.astype(np.float32)
    voxel128 = voxel128 / 255.

    mask = (voxel128[-1] > 0.9).astype(int)
    index = np.nonzero(mask)

    x = index[0].reshape(-1, 1)
    y = index[1].reshape(-1, 1)
    z = index[2].reshape(-1, 1)
    coords = np.concatenate((x, y, z), axis=1)

    r = voxel128[0][index].reshape(-1, 1)
    g = voxel128[1][index].reshape(-1, 1)
    b = voxel128[2][index].reshape(-1, 1)
    colors = np.concatenate((r, g, b), axis=1)
    voxel128 = [coords, colors]
    ######################################################################################################

    view_ids = ['{}.png'.format(i) for i in range(0, 12)]
    pre_path = '../../224/{}/{}/'.format(category, model_id)
    img_paths = [pre_path+view_id for view_id in view_ids]
    images = []
    for img_path in img_paths:
        img = np.array(cv.imread(img_path)).astype(np.float32) / 255.
        images.append(img)
    images = np.array(images).transpose(0, 3, 1, 2)

    np.savez_compressed(save_dir + '/' + model_id + '.npz', voxel32=voxel32, voxel64=voxel64, voxel128=voxel128, images=images)

with Pool(14) as p:
    r = list(tqdm(p.imap(save_npz, train_dataset), total=len(train_dataset))) # 59777
    r = list(tqdm(p.imap(save_npz, val_dataset), total=len(val_dataset))) # 7435
    r = list(tqdm(p.imap(save_npz, test_dataset), total=len(test_dataset))) # 7452
