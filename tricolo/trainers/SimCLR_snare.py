import os
import sys
import clip
import json
import pickle
import shutil
import logging
import datetime
import numpy as np
from tqdm import tqdm
import yaml

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import Compose, Normalize

from tricolo.loss.nt_xent_snare import NTXentLoss
from tricolo.loss.triplet_loss import TripletLoss
from tricolo.models.retrieval_model import ModelCLR
from tricolo.metrics.eval_retrieval import compute_metrics # change to eval_retrieval_t2s when calcualting shape2text

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

def _save_config_file(model_checkpoints_folder, config):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
    with open(os.path.join(model_checkpoints_folder, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

class SimCLR(object):
    def __init__(self, dataset, config, param_id=-1, gpu_id=-1):
        self.config = config
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.config['train']:
            log_dir = 'logs/retrieval/' + datetime.datetime.now().strftime("%b%d_%H-%M-%S")
            if param_id != -1:
                log_dir += "_CFG_"+str(param_id)
            if gpu_id != -1:
                log_dir += "_GPU_"+str(gpu_id)
            print(f"log directory is at {log_dir}\n")
            self.writer =  SummaryWriter(log_dir)
        self.dataset = dataset

        if config['loss']['type'] == 'ntxent':
            print("use ntxent")
            self.loss_fn= NTXentLoss(self.device, config['batch_size'], **config['loss']['ntxent'])
        elif config["loss"]["type"] == "triplet":
            print("use triplet")
            self.loss_fn = TripletLoss(self.device, config['batch_size'], **config['loss']['triplet'])  
        
        self.use_voxel = config['model']['use_voxel']
        self.tri_modal = config['model']['tri_modal']
        self.num_images = config['model']['num_images']
        self.sparse_model = config['model']['sparse_model']
        self.multiplier = 12 // self.num_images

        if not self.config['train']:
            if self.config['CLIP']:
                model, preprocess = clip.load("V iT-B/32", device=self.device, jit=False)
                self.clip_model = model
                self.clip_preprocess = preprocess
        
        self.info = config["info"]
        with open(self.info, 'r') as f:
            inputs_list = json.load(f)
        self.idx_to_word = inputs_list['word2idx']
        self.word_to_idx = inputs_list['idx2word']
    
    def train_pair(self):
        train_loader, val_loader, _ = self.dataset.get_data_loaders()

        model = ModelCLR(self.config["dset"], **self.config["model"])
        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)
        print("use ", torch.cuda.device_count(), " GPUs")
        model.to(self.device)
        if self.config["log_dir"] != 'None':
            model = self._load_pre_trained_weights(model, self.config["log_dir"])
            
        optimizer = torch.optim.Adam(model.parameters(), eval(self.config['learning_rate']), weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1) # T_max=len(train_loader) before

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder, self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_acc = -np.inf
        loss_fn = torch.nn.CrossEntropyLoss()

        print(f'Training...')
        valid_acc = self._validate(model, val_loader, n_iter)
        print(f'First result on validation: {valid_acc}')
        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            
            for entry_ids, arrays, images_1, images_2, voxels, voxels_2 in tqdm(val_loader): #TODO: train_loader 
                entry_ids = entry_ids.to(self.device)

                xls = arrays
                xls = xls.to(self.device)

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    voxels_2 = voxels_2.to(self.device)
                    images = images.to(self.device)
                    images_2 = images_2.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                else:
                    images_1 = images_1.to(self.device)
                    images_2 = images_2.to(self.device)

                optimizer.zero_grad()

                images_1 = images_1.reshape(-1, images_1.shape[2], images_1.shape[3], images_1.shape[4])
                z_images_1 = model.image_encoder(images_1)
                images_2 = images_2.reshape(-1, images_2.shape[2], images_2.shape[3], images_2.shape[4])
                z_images_2 = model.image_encoder(images_2)
                
                z_voxels = model.voxel_encoder(voxels)
                z_voxels_2 = model.voxel_encoder(voxels_2)

                zls = model.text_encoder(xls)
                # z_voxels, z_images, zls = model(voxels, images, xls)
                zls_img1 = torch.bmm(zls.view(zls.shape[0],1,zls.shape[1]), z_images_1.view(z_images_1.shape[0],z_images_1.shape[1], 1))
                zls_img2 = torch.bmm(zls.view(zls.shape[0],1,zls.shape[1]), z_images_2.view(z_images_2.shape[0],z_images_2.shape[1], 1))

                # loss = self.loss_fn(z_images, zls, labels) # add lables in loss function
                pred = torch.cat((zls_img1.squeeze(dim=-1), zls_img2.squeeze(dim=-1)), dim=1)
                loss = loss_fn(pred, entry_ids)
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
    
                # ## debug
                # if n_iter == 1:
                #     break

                n_iter += 1
            
            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))
                
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_acc = self._validate(model, valid_loader, n_iter)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print("best_valid_acc: ", best_valid_acc)
                self.writer.add_scalar('validation_acc', valid_acc, global_step=n_iter)
                valid_n_iter += 1

            if epoch_counter >= 2:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
    
    def train_batch(self):
        train_loader, valid_loader, _ = self.dataset.get_data_loaders()

        model = ModelCLR(self.config["dset"], **self.config["model"])
        #if torch.cuda.device_count() > 1:
        #    model = torch.nn.DataParallel(model)
        print("use ", torch.cuda.device_count(), " GPUs")
        model.to(self.device)
        if self.config["log_dir"] != 'None':
            model = self._load_pre_trained_weights(model, self.config["log_dir"])
            
        optimizer = torch.optim.Adam(model.parameters(), eval(self.config['learning_rate']), weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1) # T_max=len(train_loader) before

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder, self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_acc = -np.inf

        print(f'Training...')
        # valid_acc = self._validate(model, valid_loader, n_iter)
        # print(f'Run on Validation split before training: {valid_acc}')
    
        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            
            for labels, arrays, images, voxels in tqdm(train_loader):
                xls = arrays.to(self.device)
                

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    images = images.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                else:
                    images = images.to(self.device) # 12, 8, 3, 128, 128

                optimizer.zero_grad()

                # handle changing number of images:
                if self.config["model"]["num_images"] == 1:
                    new_images_list = []
                    for i in range(self.config["batch_size"]):
                        choose_idx = np.random.choice(range(8))
                        choose_img = images[i][choose_idx]
                        new_images_list.append(choose_img)
                    images = torch.stack(new_images_list) # 12, 3, 128, 128 
                    images = torch.unsqueeze(images, dim=1)
                if self.config["model"]["num_images"] == 2:
                    new_images_list = []
                    for i in range(self.config["batch_size"]):
                        two_images_list = []
                        choose_idx = np.random.choice(range(8))
                        choose_img = images[i][choose_idx]
                        two_images_list.append(choose_img)

                        while True:
                            choose_another = np.random.choice(range(8))
                            if choose_another != choose_idx:
                                break 
                        choose_anoimg = images[i][choose_another]
                        two_images_list.append(choose_anoimg)
                        
                        two_images = torch.stack(two_images_list)
                        new_images_list.append(two_images)

                    images = torch.stack(new_images_list) # 12, 2, 3, 128, 128 
               
                z_voxels, z_images, zls = model(voxels, images, xls)
                
                if self.tri_modal:
                    loss = self.loss_fn(z_images, z_voxels, labels) + self.loss_fn(z_voxels, zls, labels) + self.loss_fn(z_images, zls, labels)
                elif self.use_voxel:
                    loss = self.loss_fn(z_voxels, zls, labels)
                else:
                    loss = self.loss_fn(z_images, zls, labels)

                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
    
                # ## debug
                if n_iter == 1:
                    break

                n_iter += 1
            
            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))
                
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_acc = self._validate(model, valid_loader, n_iter)
                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                    print("best_valid_acc: ", best_valid_acc)
                self.writer.add_scalar('validation_acc', valid_acc, global_step=n_iter)
                valid_n_iter += 1

            if epoch_counter >= 2:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model, log_dir):
        try:
            checkpoints_folder = os.path.join(log_dir, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth')) # TODO:
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained model with success. Loading from {checkpoints_folder}")
        except FileNotFoundError:
            print("Pre-trained weights not found.")
            raise
        return model

    def _validate(self, model, valid_loader, n_iter):
        
        model.eval()
        with torch.no_grad():
            # calculate accuracy
            counter = 0
            valid_acc = 0.0
            print(f'Validation step')
            for entry_ids, arrays, images, images_2, voxels, voxels_2 in tqdm(valid_loader):
                xls = arrays
                xls = xls.to(self.device)

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    voxels_2 = voxels_2.to(self.device)
                    images = images.to(self.device)
                    images_2 = images_2.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                    voxels_2 = voxels_2.to(self.device)
                else:
                    images = images.to(self.device)
                    images_2 = images_2.to(self.device)

                zls = model.text_encoder(xls)
                
                # handle chaning number for images1
                if self.config["model"]["num_images"] == 1:
                    new_images_list = []
                    for i in range(self.config["batch_size"]):
                        choose_idx = np.random.choice(range(8))
                        choose_img = images[i][choose_idx]
                        new_images_list.append(choose_img)
                    images = torch.stack(new_images_list) # 12, 3, 128, 128 
                    images = torch.unsqueeze(images, dim=1)
                if self.config["model"]["num_images"] == 2:
                    new_images_list = []
                    for i in range(self.config["batch_size"]):
                        two_images_list = []
                        choose_idx = np.random.choice(range(8))
                        choose_img = images[i][choose_idx]
                        two_images_list.append(choose_img)

                        while True:
                            choose_another = np.random.choice(range(8))
                            if choose_another != choose_idx:
                                break 
                        choose_anoimg = images[i][choose_another]
                        two_images_list.append(choose_anoimg)
                        
                        two_images = torch.stack(two_images_list)
                        new_images_list.append(two_images)

                    images = torch.stack(new_images_list) # 12, 2, 3, 128, 128 
                    
                images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
                z_images= model.image_encoder(images)
                
                
                # images2: handle chaning number
                if self.config["model"]["num_images"] == 1:
                    new_images_list = []
                    for i in range(self.config["batch_size"]):
                        choose_idx = np.random.choice(range(8))
                        choose_img = images_2[i][choose_idx]
                        new_images_list.append(choose_img)
                    images = torch.stack(new_images_list) # 12, 3, 128, 128 
                    images = torch.unsqueeze(images, dim=1)
                if self.config["model"]["num_images"] == 2:
                    new_images_list = []
                    for i in range(self.config["batch_size"]):
                        two_images_list = []
                        choose_idx = np.random.choice(range(8))
                        choose_img = images_2[i][choose_idx]
                        two_images_list.append(choose_img)

                        while True:
                            choose_another = np.random.choice(range(8))
                            if choose_another != choose_idx:
                                break 
                        choose_anoimg = images_2[i][choose_another]
                        two_images_list.append(choose_anoimg)
                        
                        two_images = torch.stack(two_images_list)
                        new_images_list.append(two_images)

                    images_2 = torch.stack(new_images_list) # 12, 2, 3, 128, 128 
                 
                images_2 = images_2.reshape(-1, images_2.shape[2], images_2.shape[3], images_2.shape[4])
                z_images_2= model.image_encoder(images_2)

                z_voxels = model.voxel_encoder(voxels)
                z_voxels_2 = model.voxel_encoder(voxels_2)


                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_voxels_2 = F.normalize(z_voxels_2, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    z_images_2 = F.normalize(z_images_2, dim=1)
                    acc = self.cal_acc(entry_ids, zls, [z_images, z_images_2, z_voxels, z_voxels_2])
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_voxels_2 = F.normalize(z_voxels_2, dim=1)
                    acc = self.cal_acc(entry_ids, zls, [z_voxels, z_voxels_2])
                else:
                    z_images = F.normalize(z_images, dim=1)
                    z_images_2 = F.normalize(z_images_2, dim=1)
                    acc = self.cal_acc(entry_ids, zls, [z_images, z_images_2])
                
                valid_acc += acc
                counter += 1
            valid_acc /= counter
        model.train()
        return valid_acc

    def cal_acc(self, entry_ids, zls, shape_list):
        bs = self.config['batch_size']
        corr = 0
        if self.tri_modal:
            z_images, z_images_2, z_voxels, z_voxels_2 = shape_list 
            for i in range(bs):
                entry_id = entry_ids[i]
                score1 = torch.dot(zls[i], z_images[i]+z_voxels[i])
                score2 = torch.dot(zls[i], z_images_2[i]+z_voxels_2[i])
                pred = 0 if score1 > score2 else 1
                if pred == entry_id:
                    corr += 1
        elif self.use_voxel:
            z_voxels, z_voxels_2 = shape_list
            for i in range(bs):
                entry_id = entry_ids[i]
                score1 = torch.dot(zls[i], z_voxels[i])
                score2 = torch.dot(zls[i], z_voxels_2[i])
                pred = 0 if score1 > score2 else 1
                if pred == entry_id:
                    corr += 1
        else:
            z_images, z_images_2 = shape_list
            for i in range(bs):
                entry_id = entry_ids[i]
                score1 = torch.dot(zls[i], z_images[i])
                score2 = torch.dot(zls[i], z_images_2[i])
                pred = 0 if score1 > score2 else 1
                if pred == entry_id:
                    corr += 1
        
        acc = corr/bs 
        return acc 
        
    
    def save_output(self, log_dir, split='test'):
        with torch.no_grad():
            train_loader, val_loader, test_loader = self.dataset.get_data_loaders()

            model = ModelCLR(self.config["dset"], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model, log_dir)
            model.eval()

            model_test_folder = os.path.join(log_dir, split)
            _save_config_file(model_test_folder, self.config)

            print('Testing...')

            modelids = []
            text_embeds = []
            shape_embeds = []
            category_list = []
            all_caption_indices = []
            cur_loader = val_loader if split=='validate' else test_loader
            for arrays, images in tqdm(cur_loader): 
                xls = arrays
                xls = xls.to(self.device)

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    images = images.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                else:
                    images = images.to(self.device)

                #TODO:
                voxels = 0
                z_voxels, z_images, zls = model(voxels, images, xls)
                zls = F.normalize(zls, dim=1)
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    shape_embeds.append((z_images+z_voxels).detach().cpu().numpy())
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    shape_embeds.append(z_voxels.detach().cpu().numpy())
                else:
                    z_images = F.normalize(z_images, dim=1)
                    shape_embeds.append(z_images.detach().cpu().numpy())

                text_embeds.append(zls.detach().cpu().numpy())

                caption_indices = arrays.detach().cpu().numpy()
                for cap in caption_indices:
                    all_caption_indices.append(cap)

            all_text = np.vstack(text_embeds)
            all_shape = np.vstack(shape_embeds)
            assert all_text.shape[0] == all_shape.shape[0]

            tuples = []
            embeddings_dict = {}
            for i in range(all_text.shape[0]):
                new_tup = (all_caption_indices[i], category_list[i], modelids[i], all_text[i], all_shape[i]) 
                tuples.append(new_tup)
            embeddings_dict['caption_embedding_tuples'] = tuples
            save_output_path = os.path.join(model_test_folder, 'output.p')
            with open(save_output_path, 'wb') as f:
                pickle.dump(embeddings_dict, f)
                print(f"saved output dict to {save_output_path}")
        return save_output_path

    def test(self, log_dir):
        model_test_folder = os.path.join(log_dir, 'test')
        embeddings_path = self.save_output(log_dir, 'test')
        metric = 'cosine'
        dset = self.config['dset']

        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        render_dir = os.path.join(os.path.dirname(embeddings_path), 'nearest_neighbor_renderings')
        pr_at_k = compute_metrics(self.info, dset, embeddings_dict, model_test_folder, metric, concise=render_dir)
        return pr_at_k
    
    def validate(self, log_dir):
        model_test_folder = os.path.join(log_dir, 'validate')

        embeddings_path = self.save_output(log_dir, 'validate')
        metric = 'cosine'
        dset = self.config['dset']

        with open(embeddings_path, 'rb') as f:
            embeddings_dict = pickle.load(f)

        render_dir = os.path.join(os.path.dirname(embeddings_path), 'nearest_neighbor_renderings')
        pr_at_k = compute_metrics(self.info, dset, embeddings_dict, model_test_folder, metric, concise=render_dir)
        return pr_at_k
    

   