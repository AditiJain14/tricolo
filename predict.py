import json
import argparse
from tqdm import tqdm

import torch

from tricolo.trainers.SimCLR import SimCLR
from tricolo.dataloader.dataset_wrapper import DataSetWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--config_file", default='tricolo/configs/retrieval_shapenet.yaml', type=str, help="Path to config file")
args = parser.parse_args()


def main(load_dir, customized_sentence):
    with open(load_dir + '/checkpoints/config.json', 'r') as f:
        config = json.load(f)
    config['train'] = False
    if not 'CLIP' in config.keys(): 
        config['CLIP'] = False
    config['log_dir'] = load_dir
    config['batch_size'] = 64


    dataset = DataSetWrapper(config['dset'], config['batch_size'], config['train'], **config['dataset'])
    simclr = SimCLR(dataset, config)
    
    simclr.predict(config['log_dir'], customized_sentence)

# Nov09_21-45-44_Cfg0_GPU0 are b128v64i128

if __name__ == "__main__":
    load_dir = 'logs/retrieval/v64i128b128/Nov05_07-16-16-0'
    customized_sentence = "circular table with glass" # Change here as you want

    main(load_dir, customized_sentence)
    torch.cuda.empty_cache()
    
    # The retrieved shape would be stored in load_dir 