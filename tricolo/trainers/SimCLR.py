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
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from tricolo.loss.nt_xent import NTXentLoss, NTXentLoss_neg
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
                log_dir += "_CFG"+str(param_id)
            if gpu_id != -1:
                log_dir += "_GPU"+str(gpu_id)
            print(f"log directory is at {log_dir}\n")
            self.writer =  SummaryWriter(log_dir)
        self.dataset = dataset

        if config['loss']['type'] == 'ntxent':
            print("use ntxent")
            self.loss_fn= NTXentLoss(self.device, config['batch_size'], **config['loss']['ntxent'])
        elif config["loss"]["type"] == 'ntxent_neg':
            print("use ntxent with negative sampling")
            self.loss_fn= NTXentLoss_neg(self.device, config['batch_size'], **config['loss']['ntxent_neg'])
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
        
        self.info = os.path.join(config['info'], config['dset'], config['dset']+'.json')
        with open(self.info, 'r') as f:
            inputs_list = json.load(f)
        self.idx_to_word = inputs_list['idx_to_word']
        self.word_to_idx = inputs_list['word_to_idx']
    
    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        model = ModelCLR(self.config["dset"], **self.config["model"])
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        print("use ", torch.cuda.device_count(), " GPUs")
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), eval(self.config['learning_rate']), weight_decay=eval(self.config['weight_decay']))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1) # T_max=len(train_loader) before

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder, self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        print(f'Training...')
        for epoch_counter in range(self.config['epochs']):
            print(f'Epoch {epoch_counter}')
            for model_id, voxels, categories, arrays, images in tqdm(train_loader):
                xls = arrays
                xls = xls.to(self.device)

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    images = images[:, ::self.multiplier]
                    images = images.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                else:
                    images = images[:, ::self.multiplier]
                    images = images.to(self.device)

                optimizer.zero_grad()
                z_voxels, z_images, zls = model(voxels, images, xls)
                # print("Outside: input size", images.size(),
                #         "output_size", z_images.size())
                zls = F.normalize(zls, dim=1)
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.loss_fn(z_images, z_voxels) + self.loss_fn(z_voxels, zls) + self.loss_fn(z_images, zls)
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    loss = self.loss_fn(z_voxels, zls)
                else:
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.loss_fn(z_images, zls)
                loss.backward()
                optimizer.step()

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)

                n_iter += 1

            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(epoch_counter)))
                
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader, n_iter)
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            if epoch_counter >= 2:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model, log_dir):
        try:
            checkpoints_folder = os.path.join(log_dir, 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict, strict=False)
            print(f"Loaded pre-trained model with success. Loading from {checkpoints_folder}")
        except FileNotFoundError:
            print("Pre-trained weights not found.")
            raise
        return model

    def _validate(self, model, valid_loader, n_iter):
        model.eval()
        with torch.no_grad():
            counter = 0
            valid_loss = 0.0
            print(f'Validation step')
            for model_id, voxels, categories, arrays, images in tqdm(valid_loader):
                xls = arrays
                xls = xls.to(self.device)

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    images = images[:, ::self.multiplier]
                    images = images.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                else:
                    images = images[:, ::self.multiplier]
                    images = images.to(self.device)

                z_voxels, z_images, zls = model(voxels, images, xls)
                zls = F.normalize(zls, dim=1)
                if self.tri_modal:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.loss_fn(z_images, z_voxels) + self.loss_fn(z_voxels, zls) + self.loss_fn(z_images, zls)
                elif self.use_voxel:
                    z_voxels = F.normalize(z_voxels, dim=1)
                    loss = self.loss_fn(z_voxels, zls)
                else:
                    z_images = F.normalize(z_images, dim=1)
                    loss = self.loss_fn(z_images, zls)

                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        model.train()
        return valid_loss
    
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
            for modelid, voxels, categories, arrays, images in tqdm(cur_loader): 
                xls = arrays
                xls = xls.to(self.device)

                if self.tri_modal:
                    voxels = voxels.to(self.device)
                    images = images[:, ::self.multiplier]
                    images = images.to(self.device)
                elif self.use_voxel:
                    voxels = voxels.to(self.device)
                else:
                    images = images[:, ::self.multiplier]
                    images = images.to(self.device)

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
                modelids.extend(list(modelid))
                category_list.extend(list(categories))

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

    def save_output_clip(self, log_dir):
        with torch.no_grad():
            train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

            model_test_folder = os.path.join(log_dir, 'test')

            print('Testing...')

            modelids = []
            text_embeds = []
            shape_embeds = []
            category_list = []
            all_caption_indices = []
            for modelid, voxels, categories, arrays, images in tqdm(test_loader):
                images = images.to(self.device).reshape(images.shape[0]*12, images.shape[2], images.shape[3], images.shape[4])
                text = clip.tokenize(xls).to(self.device)

                def _transform():
                    return Compose([
                        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),])
                preprocess = _transform()
                images = preprocess(images)

                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images).reshape(-1, 12, 512)
                    image_features = torch.mean(image_features, 1)
                    text_features = self.clip_model.encode_text(text)

                if self.config["model"]["bert_base_model"]=="bert-base-uncased":
                    xls = self.tokenizer(list(xls), return_tensors="pt", padding=True, truncation=self.truncation)
                if self.config["model"]["bert_base_model"]=="BiGRU":
                    xls = arrays
                xls = xls.to(self.device)

                shape_embeds.append(image_features.detach().cpu().numpy())
                text_embeds.append(text_features.detach().cpu().numpy())
                modelids.extend(list(modelid))
                category_list.extend(list(categories))

                if self.config["model"]["bert_base_model"]=="bert-base-uncased":
                    caption_indices = xls['input_ids'].detach().cpu().numpy()
                if self.config["model"]["bert_base_model"]=="BiGRU":
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
    

    def predict(self, log_dir, custom_sen):
        '''
        sens = ["dark brown colored wooden table",
        "circular table with glass",
        "rectangular table with glass",
        "a fashion chair",
        "coffee table",
        "chair with arm",
        "chair without arm",
        "rectangular table with 1 shelf",
        "green table",
        ]
        
        '''
        with torch.no_grad():
            model = ModelCLR(self.config["dset"], **self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model, log_dir)
            model.eval()

            predict_folder = os.path.join(log_dir, 'predict')
            if not os.path.exists(predict_folder):
                os.makedirs(predict_folder)

            print('Predicting...')

            text_embeds = []
            
            cur = [self.word_to_idx[w] for w in custom_sen.split()]
            length = len(cur)
            cur.extend([0]*(96-length))
            arrays = torch.from_numpy(np.array(cur))
            xls = arrays
            xls = xls.to(self.device)
            xls = xls.unsqueeze(0)
            zls = model.text_encoder(xls)
            zls = F.normalize(zls, dim=1)
            text_embeds = zls.detach().cpu().numpy()
            
            # get tuples
            with open(os.path.join(log_dir, "test/output.p"), 'rb') as f:
                embeddings_dict = pickle.load(f)
            tuples = embeddings_dict['caption_embedding_tuples']
            # calculate the nearest shape
            shape_embeddings_list = []
            model_id_to_label = {}
            label_to_model_id = {}
            label_counter = 0
            
            for idx, caption_tuple in enumerate(tuples):
                # Parse caption tuple
                caption, category, model_id, text_embedding, shape_embedding = caption_tuple
                if model_id not in model_id_to_label:
                    model_id_to_label[model_id] = label_counter
                    label_to_model_id[label_counter] = model_id

                    shape_embeddings_list.append(shape_embedding)
                    label_counter += 1

            shape_embeddings_matrix = np.vstack(shape_embeddings_list) # 1492,512
            
            unnormalized_similarities = np.dot(text_embeds, shape_embeddings_matrix.T)
            sort_indices = np.argsort(unnormalized_similarities, axis=1)
            nearest = sort_indices[:, -1][0] # could be changed to sort_indices[:-20] to see more predicted shapes
            retrieved_modelid = label_to_model_id[nearest]
 
            nrrd_path = os.path.join(self.config["dataset"]["voxel_root_dir"], retrieved_modelid, retrieved_modelid+'.nrrd')
            img_path = os.path.join(self.config["dataset"]["voxel_root_dir"], retrieved_modelid, retrieved_modelid+'.png')
            shutil.copyfile(img_path, os.path.join(predict_folder, retrieved_modelid+'.png'))