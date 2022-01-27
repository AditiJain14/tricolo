import torch
import torch.nn as nn
import torch.nn.functional as F

from tricolo.models.models import cnn_encoder, cnn_encoder32, SVCNN, MVCNN 

class ModelCLR(nn.Module):
    def __init__(self, dset, out_dim, sparse_model, use_voxel, tri_modal, num_images, image_cnn, pretraining, vocab_size):
        super(ModelCLR, self).__init__()

        self.ef_dim = 32
        self.z_dim = 512
        self.dset = dset
        self.out_dim = out_dim
        self.cnn_name = image_cnn
        self.use_voxel = use_voxel
        self.tri_modal = tri_modal
        self.num_images = num_images
        self.pretraining = pretraining
        self.sparse_model = sparse_model

        self.text_model = self._get_text_basemodel()

        self.voxel_model, self.voxel_fc, self.image_model, self.image_fc = self._get_res_basemodel()

        self.embedding_layer = nn.Embedding(vocab_size, 256, padding_idx=0) 
        self.fc = nn.Linear(256, out_dim)

    def _get_res_basemodel(self):
        voxel_model = None
        voxel_fc = None
        image_model = None
        image_fc = None
        if self.dset == 'shapenet':
            if self.tri_modal:
                voxel_model = cnn_encoder(self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))

                svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                print('Tri-Modal Voxel, Image {} {} views, Text'.format(self.cnn_name, self.num_images))
            elif self.use_voxel:
                voxel_model = cnn_encoder(self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                print('Bi-Modal Voxel, Text')
            else:
                svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                print('Bi-Modal Image {} {} views, Text'.format(self.cnn_name, self.num_images))
        elif self.dset == 'primitives':
            if self.tri_modal:
                raise('Implement Other Dataset')
                # voxel_model = cnn_encoder32(self.ef_dim, self.z_dim)
                # voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))

                # svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                # image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                # image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                # print('Tri-Modal Voxel, Image {} {} views, Text'.format(self.cnn_name, self.num_images))
            elif self.use_voxel:
                voxel_model = cnn_encoder32(self.ef_dim, self.z_dim)
                voxel_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                print('Bi-Modal Voxel, Text')
            else:
                raise('Implement Other Dataset')
                # svcnn = SVCNN(self.z_dim, pretraining=self.pretraining, cnn_name=self.cnn_name)
                # image_model = MVCNN(self.z_dim, svcnn, cnn_name=self.cnn_name, num_views=self.num_images)
                # image_fc = nn.Sequential(nn.Linear(self.z_dim,self.out_dim),nn.ReLU(),nn.Linear(self.out_dim,self.out_dim))
                # print('Bi-Modal Image {} {} views, Text'.format(self.cnn_name, self.num_images))
        else:
            raise('Implement Other Dataset')
        return voxel_model, voxel_fc, image_model, image_fc

    def _get_text_basemodel(self):
        model = nn.GRU(input_size=256, hidden_size=128, num_layers=1, bidirectional=True) 
        return model

    
    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html
        """
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def voxel_encoder(self, xis):
        h = self.voxel_model(xis)
        h.squeeze()
        x = self.voxel_fc(h)
        return x

    def image_encoder(self, xis):
        h = self.image_model(xis)
        h.squeeze()
        x = self.image_fc(h)
        return x

    def text_encoder(self, encoded_inputs):
        embed_inputs = self.embedding_layer(encoded_inputs) # N, L, Hin
        embed_inputs = torch.transpose(embed_inputs, 0, 1) # L, N, Hin
        N = embed_inputs.shape[1]
        """
        Randn init for BiGRU good?
        """
        h0 = torch.zeros(2, N, 128).cuda()
        # h0 = torch.randn(2, N, 128).cuda()

        self.text_model.flatten_parameters() # Handle this problem: RNN module weights are not part of single contiguous chunk of memory. This means they need to be compacted at every call, possibly greately increasing memory usage. 
        output, hidden = self.text_model(embed_inputs, h0) # hn: (D*layer, N, Hout), (2, N, 128)
        # concate hidden in two directions (N, 128*2)
        out_emb = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))) # (N, 256)

        return out_emb

    def forward(self, voxels, images, encoded_inputs):
        z_voxels = None
        z_images = None
        if self.tri_modal:
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            z_voxels = self.voxel_encoder(voxels)
            z_images = self.image_encoder(images)
        elif self.use_voxel:
            z_voxels = self.voxel_encoder(voxels)
        else:
            images = images.reshape(-1, images.shape[2], images.shape[3], images.shape[4])
            z_images = self.image_encoder(images)

        zls = self.text_encoder(encoded_inputs)

        # print("\tIn Model: input size", images.size(),
        #       "output size", z_images.size())

        return z_voxels, z_images, zls
