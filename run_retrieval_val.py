import os
import json
import argparse
from numpy import *

import torch

from tricolo.trainers.SimCLR import SimCLR
from tricolo.dataloader.dataset_wrapper import DataSetWrapper

parser = argparse.ArgumentParser()
parser.add_argument("--validate", action='store_true', help="evaluate model on VALIDATION set")
parser.add_argument("--test", action='store_true', help="evaluate model on TEST set")
args = parser.parse_args()

def main(load_dir):
    with open(load_dir + '/checkpoints/config.json', 'r') as f:
        config = json.load(f)
    print(config['model'])
    config['train'] = False
    if not 'CLIP' in config.keys(): 
        config['CLIP'] = False
    config['log_dir'] = load_dir

    dataset = DataSetWrapper(config['dset'], config['batch_size'], config['train'], **config['dataset'])
    simclr = SimCLR(dataset, config)

    if config['CLIP']:
        simclr.test('./logs/CLIP')
    if args.validate:
        pr_at_k = simclr.validate(config['log_dir'])
    if args.test:
        if config['CLIP']:
            pr_at_k = simclr.test('./logs/CLIP')
        else:
            pr_at_k = simclr.test(config['log_dir'])
    return pr_at_k



if __name__ == "__main__":
    # load_dirs = ['./logs/retrieval/i128b128/Nov04_03-39-03', './logs/retrieval/i128b128/Nov04_06-47-54', './logs/retrieval/i128b128/Nov04_09-58-07']
    # load_dirs = ['./logs/retrieval/v64b128/Nov04_03-17-07-2', './logs/retrieval/v64b128/Nov04_03-17-07-3']
    # load_dirs = ['./logs/retrieval/v64i128b128/Nov05_07-16-16-0', './logs/retrieval/v64i128b128/Nov05_07-16-16-1', './logs/retrieval/v64i128b128/Nov05_07-16-16-2', './logs/retrieval/v64i128b128/Nov05_07-16-16-3']
    
    # load_dirs = ['logs/retrieval/Nov09_13-49-45_Cfg10_GPU0', 'logs/retrieval/Nov09_13-49-45_Cfg10_GPU1', 'logs/retrieval/Nov09_13-49-45_Cfg10_GPU2'] # triplet: tri image+voxel
    # load_dirs = ['logs/retrieval/Nov09_14-46-46_Cfg8_GPU0', 'logs/retrieval/Nov09_14-46-46_Cfg8_GPU1', 'logs/retrieval/Nov09_14-46-46_Cfg8_GPU2'] # triplet: bi voxel
    load_dirs = ['logs/retrieval/triplet/biI/Nov09_12-06-17_Cfg9_GPU0', 'logs/retrieval/triplet/biI/Nov09_12-06-17_Cfg9_GPU1', 'logs/retrieval/triplet/biI/Nov09_12-06-17_Cfg9_GPU2'] # triplet: bi image
    
    # model_class = "./logs/retrieval/v64i128b128"
    # load_dirs = [os.path.join(model_class, fl) for fl in os.listdir(model_class)]
    rr1 = []
    rr5 = []
    ndcg5 = []
    mrr = []
    for load_dir in load_dirs:
        pr_at_k = main(load_dir)
        print(pr_at_k.recall_rate[0], pr_at_k.recall_rate[4], pr_at_k.ndcg[4], pr_at_k.r_rank)
        rr1.append(pr_at_k.recall_rate[0])
        rr5.append(pr_at_k.recall_rate[4])
        ndcg5.append(pr_at_k.ndcg[4])
        mrr.append(pr_at_k.r_rank)
        torch.cuda.empty_cache()
    
    print(mean(rr1), mean(rr5), mean(ndcg5), mean(mrr))
    print(std(rr1), std(rr5), std(ndcg5), std(mrr))
