import json
import argparse
from tqdm import tqdm

import torch

from tricolo.trainers.SimCLR import SimCLR
from tricolo.dataloader.dataset_wrapper import DataSetWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--load_dir", default="logs/retrieval/v64i128b128/Nov05_07-16-16-0", type=str, help="Path to checkpoint file")
parser.add_argument("--customized_sentence", default="circular_table_with_glass", type=str, help="Your customized sentence")
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
    sen = " ".join(args.customized_sentence.split("_"))
    main(args.load_dir, sen)
    torch.cuda.empty_cache()
    
    # The retrieved shape would be stored in load_dir 