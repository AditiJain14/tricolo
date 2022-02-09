import json
import argparse
from numpy import *

from tricolo.trainers.SimCLR_snare import SimCLR
from tricolo.dataloader.dataset_wrapper import DataSetWrapper_snare

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

    dataset = DataSetWrapper_snare(config['batch_size'],config['train'], **config['dataset'])
    simclr = SimCLR(dataset, config)


    if args.validate:
        acc = simclr.validate(config['log_dir'])
    if args.test:
        acc = simclr.test(config['log_dir'])
    return acc



if __name__ == "__main__":
    # load_dirs = ['logs/retrieval/Jan27_23-03-07_CFG_snare_img_2'] # image 2views
    load_dirs = ['logs/retrieval/Jan30_15-35-53_CFG_snare_img_2'] # image fixed 2 views 
    # load_dirs = ['logs/retrieval/Jan27_18-15-37_CFG_snare_img'] # image 8views
    # load_dirs = ['logs/retrieval/Jan28_20-18-30_CFG_snare_vox'] # voxel
    # load_dirs = ['logs/retrieval/Jan28_16-49-10_CFG_snare_tri'] # trimodal
    # load_dirs = ['logs/retrieval/Jan31_11-38-55_CFG_snare_img_2']
    # load_dirs = ['logs/retrieval/Jan31_04-29-44_CFG_snare_vox']

    for load_dir in load_dirs:
       acc, valid_acc_visual, valid_acc_blind = main(load_dir)
       print("acc: ", acc, "acc_visual: ", valid_acc_visual, "acc_blind: ", valid_acc_blind)
       
