import yaml
import argparse
import numpy as np

import torch

from tricolo.trainers.SimCLR import SimCLR
from tricolo.dataloader.dataset_wrapper import DataSetWrapper


parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='tricolo/configs/retrieval_shapenet.yaml', type=str, help="Path to config file")
parser.add_argument("--param_id", dest='param_id', default=-1, type=int, help="search parameters on cedar")
parser.add_argument("--gpu_id", dest='gpu_id', default=-1, type=int, help="search parameters on cedar")
args = parser.parse_args()

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not dict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not (k in b):
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is dict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def main(setting=None):
    config = yaml.load(open(args.config_file, "r"), Loader=yaml.FullLoader)

    base_lr = 1e-4
    if setting is not None:
        _merge_a_into_b(setting, config)
    
    config['learning_rate'] = str((config['batch_size'] / 32) * base_lr)
    print(config['batch_size'])
    print(eval(config['learning_rate']))

    dataset = DataSetWrapper(config['dset'], config['batch_size'], config['train'], **config['dataset'])
    simclr = SimCLR(dataset, config, args.param_id, args.gpu_id)

    simclr.train()

if __name__ == "__main__":
    settings = []

    ################### trimodal ablations   ###############
    if args.param_id == 0:
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': True,
                            'tri_modal': True,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128,   
                              'voxel_size': 64} # trimodal ablations - voxel size
        settings.append(setting)

    elif args.param_id == 1:
        setting = {}
        setting['batch_size'] = 128 
        setting['model'] = {'use_voxel': True,
                            'tri_modal': True,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128,
                              'voxel_size': 128} # trimodal ablations - voxel size
        settings.append(setting)
    
    ################### voxel ablations   ###############
    elif args.param_id == 2: 
        setting = {}
        setting['batch_size'] = 32 # voxel ablations - batch size
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128,
                              'voxel_size': 64}
        settings.append(setting)
    
    elif args.param_id == 3: 
        setting = {}
        setting['batch_size'] = 64 # voxel ablations - batch size
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128,
                              'voxel_size': 64}
        settings.append(setting)
    
    elif args.param_id == 4: 
        setting = {}
        setting['batch_size'] = 128 # voxel ablations - batch size
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128,
                              'voxel_size': 64}
        settings.append(setting)
    
    elif args.param_id == 5: 
        setting = {}
        setting['batch_size'] = 256 # voxel ablations - batch size
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 64}
        settings.append(setting)

    elif args.param_id == 6: 
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 32} # voxel ablations - voxel size
        settings.append(setting)
    
    elif args.param_id == 7: 
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 64} # voxel ablations - voxel size
        settings.append(setting)
    
    elif args.param_id == 8: 
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': True,
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 128} # voxel ablations - voxel size
        settings.append(setting)

    ################### triplet ablations   ###############
    elif args.param_id == 9: 
        setting = {}
        setting['batch_size'] = 128
        setting['epochs'] = 40
        setting['model'] = {'use_voxel': True, # bimodal, voxel
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 64}
        setting['loss'] = {'type': 'triplet'}

        settings.append(setting)
    
    elif args.param_id == 10: 
        setting = {}
        setting['batch_size'] = 128
        setting['epochs'] = 40
        setting['model'] = {'use_voxel': False, # bimodal, image
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 64}
        setting['loss'] = {'type': 'triplet'}

        settings.append(setting)
    
    elif args.param_id == 11: 
        setting = {}
        setting['batch_size'] = 128
        setting['epochs'] = 40
        setting['model'] = {'use_voxel': True, 
                            'tri_modal': True, # trimodal
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128, 
                              'voxel_size': 64}
        setting['loss'] = {'type': 'triplet'}

        settings.append(setting)

    ################### images ablations   ###############
    elif args.param_id == 14: 
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': False, 
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 64,  # image ablations - image size
                              'voxel_size': 64}
        settings.append(setting)
    
    elif args.param_id == 15: 
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': False, 
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 128,  # image ablations - image size
                              'voxel_size': 64}
        settings.append(setting)
    
    elif args.param_id == 16: 
        setting = {}
        setting['batch_size'] = 128
        setting['model'] = {'use_voxel': False, 
                            'tri_modal': False,
                            'num_images': 6,
                            'image_cnn': 'resnet18',
                            'pretraining': True}
        setting['dataset'] = {'image_size': 224,  # image ablations - image size
                              'voxel_size': 64}
        settings.append(setting)

    for setting in settings:
        main(setting)
        torch.cuda.empty_cache()
