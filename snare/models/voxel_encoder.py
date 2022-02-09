# sparse convolution
import MinkowskiEngine as ME
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch

class cnn_encoder(nn.Module):
    def __init__(self, ef_dim, z_dim):
        super(cnn_encoder, self).__init__()
        self.ef_dim = ef_dim
        self.z_dim = z_dim
        '''
        self.conv_1 = nn.Conv3d(4, self.ef_dim, 3, stride=2, padding=1, bias=False)
        self.in_1 = nn.InstanceNorm3d(self.ef_dim)
        '''
        # Block 1
        self.block1 = nn.Sequential(
            ME.MinkowskiConvolution(3, self.ef_dim, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiInstanceNorm(self.ef_dim),
            ME.MinkowskiELU(),
        )
        
        
        '''
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim*2, 3, stride=1, padding=1, bias=False)
        self.in_2 = nn.InstanceNorm3d(self.ef_dim*2)
        self.pool_2 = nn.MaxPool3d(3, stride=2, padding=1)
        '''
        # Block 2
        self.block2 = nn.Sequential(
            ME.MinkowskiConvolution(self.ef_dim, self.ef_dim*2, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiInstanceNorm(self.ef_dim*2),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiELU(),
        )
        
        
        '''
        self.conv_3 = nn.Conv3d(self.ef_dim*2, self.ef_dim*4, 3, stride=1, padding=1, bias=False)
        self.in_3 = nn.InstanceNorm3d(self.ef_dim*4)
        self.pool_3 = nn.MaxPool3d(3, stride=2, padding=1)
        '''
        # Block 3
        self.block3 = nn.Sequential(
            ME.MinkowskiConvolution(self.ef_dim*2, self.ef_dim*4, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiInstanceNorm(self.ef_dim*4),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiELU(),
        )
        
        '''
        self.conv_4 = nn.Conv3d(self.ef_dim*4, self.ef_dim*8, 3, stride=1, padding=1, bias=False)
        self.in_4 = nn.InstanceNorm3d(self.ef_dim*8)
        self.pool_4 = nn.MaxPool3d(3, stride=2, padding=1)
        '''
        # Block 4
        self.block4 = nn.Sequential(
            ME.MinkowskiConvolution(self.ef_dim*4, self.ef_dim*8, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiInstanceNorm(self.ef_dim*8),
            ME.MinkowskiMaxPooling(kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiELU(),
        )
        
        '''
        self.conv_5 = nn.Conv3d(self.ef_dim*8, self.z_dim, 3, stride=2, padding=1, bias=True)
        self.in_5 = nn.InstanceNorm3d(self.z_dim)
        self.pool_5 = nn.AdaptiveAvgPool3d((2,2,2))
        self.out = nn.Linear(4096, self.z_dim)
        '''
         # Block 5
        self.block5 = nn.Sequential(
            ME.MinkowskiConvolution(self.ef_dim*8, self.z_dim, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiInstanceNorm(self.z_dim),
            ME.MinkowskiGlobalAvgPooling(), 
            ME.MinkowskiELU(),
        )
        
        self.linear = ME.MinkowskiLinear(self.z_dim, self.z_dim, bias=True)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            # if isinstance(m, ME.MinkowskiInstanceNorm):
            #     nn.init.constant_(m.bn.weight, 1)
            #     nn.init.constant_(m.bn.bias, 0)

        
        '''
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.conv_3.weight)
        nn.init.xavier_uniform_(self.conv_4.weight)
        nn.init.xavier_uniform_(self.conv_5.weight)
        nn.init.constant_(self.conv_5.bias,0)
        '''
    
    def forward(self, sinput):
        out = self.block1(sinput) # check shape 
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out) # 80, 256
        out = self.block5(out) # 2, 512
        out = self.linear(out) # 2, 512

        return out.F
    
    '''
     inputs: 32, 1, 64, 64, 64
        d_1 = self.in_1(self.conv_1(inputs))
        d_1 = F.leaky_relu(xd_1, negative_slope=0.02, inplace=True) # 32, 32, 32, 32, 32
        d_2 = self.in_2(self.conv_2(d_1))
        d_2 = F.leaky_relu(d_2, negative_slope=0.02, inplace=True) # 32, 64, 16, 16, 16
        d_2 = self.pool_2(d_2)
        d_3 = self.in_3(self.conv_3(d_2)) # 32, 128, 8, 8, 8
        d_3 = F.leaky_relu(d_3, negative_slope=0.02, inplace=True)
        d_3 = self.pool_3(d_3)
        d_4 = self.in_4(self.conv_4(d_3))
        d_4 = F.leaky_relu(d_4, negative_slope=0.02, inplace=True) # 32, 256, 4, 4, 4
        d_4 = self.pool_4(d_4)
        d_5 = self.in_5(self.conv_5(d_4)) # 32, 256, 1, 1, 1
        d_5 = F.leaky_relu(d_5, negative_slope=0.02, inplace=True)
        d_5 = self.pool_5(d_5)
        flatten = d_5.reshape(d_5.shape[0], -1) # 32, 256
        output = self.out(flatten)
        return output
    
    '''
   

if __name__=='__main__':
    category = '03001627'
    model_id = 'a6a7b00de9ccce0816a5604496f9ff11'
    path = '/local-scratch/yuer/projects/tricolo/data/all_npz/' + category + '/' + model_id + '.npz'
    data = np.load(path)
    voxel64 = data['voxel64']
    
    data_dict = {}
    coords_1, colors_1 = voxel64
    
    model_id = '5aa3f60fb8fc208d488fd132b74d6f8d'
    path = '/local-scratch/yuer/projects/tricolo/data/all_npz/' + category + '/' + model_id + '.npz'
    data = np.load(path)
    voxel64 = data['voxel64']
    
    data_dict = {}
    coords_2, colors_2 = voxel64
    
    coords, feats = ME.utils.sparse_collate([coords_1, coords_2], [colors_1, colors_2])
    input = ME.SparseTensor(feats.float(), coordinates=coords)

    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    sin = ME.SparseTensor(
        features=torch.from_numpy(colors_1),
        coordinates=torch.from_numpy(coords_1).int(),
        # device=device,
        )

    net =  cnn_encoder(32, 512)
    sout = net(input)
    
    for k, v in net.named_parameters():
        print(k, v.size())