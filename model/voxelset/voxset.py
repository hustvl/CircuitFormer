import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter
import math
import spconv.pytorch as spconv


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class PointPillarScatter(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_bev_features = 64
        self.nx, self.ny = 256, 256

    def forward(self, pillar_features, coords):
        batch_spatial_features = []
        batch_size = coords[:, 0].max().int().item() + 1
        for batch_idx in range(batch_size):
            spatial_feature = torch.zeros(
                self.num_bev_features,
                self.nx * self.ny,
                dtype=pillar_features.dtype,
                device=pillar_features.device)

            batch_mask = coords[:, 0] == batch_idx
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 1] * self.nx + this_coords[:, 2]
            indices = indices.type(torch.long)
            pillars = pillar_features[batch_mask, :]
            pillars = pillars.t()
            spatial_feature[:, indices] = pillars
            batch_spatial_features.append(spatial_feature)

        batch_spatial_features = torch.stack(batch_spatial_features, 0)
        batch_spatial_features = batch_spatial_features.view(batch_size, self.num_bev_features, self.ny, self.nx)
        spatial_features = batch_spatial_features
        return spatial_features

class MLP_VSA_Layer(nn.Module):
    def __init__(self, dim, n_latents=8):
        super(MLP_VSA_Layer, self).__init__()
        self.dim = dim
        self.k = n_latents 
        self.pre_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01),
        )

        # the learnable latent codes can be obsorbed by the linear projection
        self.score = nn.Linear(dim, n_latents)

        

        conv_dim = dim * self.k
        self.conv_dim = conv_dim
        
        # conv ffn
        self.conv_ffn = nn.Sequential(           
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False), 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, bias=False), 
            nn.BatchNorm2d(conv_dim),
            nn.ReLU(),
            nn.Conv2d(conv_dim, conv_dim, 1, 1, bias=False), 
         ) 
        
        # decoder
        self.norm = nn.BatchNorm1d(dim,eps=1e-3, momentum=0.01)
        self.mhsa = nn.MultiheadAttention(dim, num_heads=dim//4, batch_first=True)
        
    def forward(self, inp, inverse, coords, bev_shape):
        
        x = self.pre_mlp(inp)

        # encoder
        attn = torch_scatter.scatter_softmax(self.score(x), inverse, dim=0)
        dot = (attn[:, :, None] * x.view(-1, 1, self.dim)).view(-1, self.dim*self.k)
        x_ = torch_scatter.scatter_sum(dot, inverse, dim=0)

        # conv ffn
        batch_size = int(coords[:, 0].max() + 1)
        h = spconv.SparseConvTensor(F.relu(x_), coords.int(), bev_shape, batch_size).dense().squeeze(-1)
        h = self.conv_ffn(h).permute(0,2,3,1).contiguous().view(-1, self.conv_dim)
        flatten_indices = coords[:, 0] * bev_shape[0] * bev_shape[1] + coords[:, 1] * bev_shape[1] + coords[:, 2]
        h = h[flatten_indices.long(), :]
        h = h[inverse, :]
       
        # decoder
        hs = self.norm(h.view(-1,  self.dim)).view(-1, self.k, self.dim)
        hs = self.mhsa(x.view(-1, 1, self.dim), hs, hs)[0]
        hs = hs.view(-1, self.dim)
        # skip connection
        return torch.cat([inp, hs], dim=-1)
        
       

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=64, dim=128, temperature=10000):
        super().__init__()
        self.token_projection = nn.Linear(hidden_dim * 2, dim)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        

    def forward(self, pos_embed, max_len=(1, 1)):
        y_embed, x_embed = pos_embed.chunk(2, 1)
        x_max, y_max = max_len
        
        eps = 1e-6
        x_embed = x_embed / (x_max + eps) * self.scale
        y_embed = y_embed / (y_max + eps) * self.scale


        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=pos_embed.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed / dim_t
        pos_y = y_embed / dim_t


        pos_x = torch.stack((pos_x[:, 0::2].sin(),
                             pos_x[:, 1::2].cos()), dim=2).flatten(1)
        pos_y = torch.stack((pos_y[:, 0::2].sin(),
                             pos_y[:, 1::2].cos()), dim=2).flatten(1)


        pos = torch.cat((pos_y, pos_x), dim=1)

        pos = self.token_projection(pos)
        return pos
    
class VoxSeT(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_latents = [8, 8, 8, 8]
        self.input_dim = 16
        self.output_dim = 64
        num_point_features = 9
        self.input_embed = MLP(num_point_features, 16, self.input_dim, 2)
       
        self.pe0 = PositionalEncodingFourier(64, self.input_dim)
        self.pe1 = PositionalEncodingFourier(64, self.input_dim * 2)
        self.pe2 = PositionalEncodingFourier(64, self.input_dim * 4)
        self.pe3 = PositionalEncodingFourier(64, self.input_dim * 8)
    
        self.mlp_vsa_layer_0 = MLP_VSA_Layer(self.input_dim * 1, self.num_latents[0])
        self.mlp_vsa_layer_1 = MLP_VSA_Layer(self.input_dim * 2, self.num_latents[1])
        self.mlp_vsa_layer_2 = MLP_VSA_Layer(self.input_dim * 4, self.num_latents[2])
        self.mlp_vsa_layer_3 = MLP_VSA_Layer(self.input_dim * 8, self.num_latents[3])
        self.PointPillarScatter = PointPillarScatter()


        self.post_mlp = nn.Sequential(
            nn.Linear(self.input_dim * 16, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim, eps=1e-3, momentum=0.01)
        )


        self.register_buffer('voxel_size', torch.FloatTensor([1,1]))




        self.register_buffer('voxel_size_02x', torch.FloatTensor([2, 2]))
        self.register_buffer('voxel_size_04x', torch.FloatTensor([4, 4]))
        self.register_buffer('voxel_size_08x', torch.FloatTensor([8, 8]))

        self.grid_size = [256, 256]
        a, b = self.grid_size
        self.grid_size_02x = [a // 2, b // 2]
        self.grid_size_04x = [a // 4, b // 4]
        self.grid_size_08x = [a // 8, b // 8]

        # self.apply(self._kaiming_normal_init)
        
    def get_output_feature_dim(self):
        return self.output_dim


    def forward(self, points):

        points_offsets = points[:, 1:3]  # x y


        pe_raw = points_offsets
        coords01x = points[:, :3].clone() # index x y
        coords01x[:, 1:3] = (points_offsets - 1e-5)// self.voxel_size       #Avoid subscripts out of bounds
        coords01x, inverse01x = torch.unique(coords01x, return_inverse=True, dim=0)

        coords02x = points[:, :3].clone()
        coords02x[:, 1:3] = (points_offsets - 1e-5) // self.voxel_size_02x
        coords02x, inverse02x = torch.unique(coords02x, return_inverse=True, dim=0)

        coords04x = points[:, :3].clone()
        coords04x[:, 1:3] = (points_offsets - 1e-5) // self.voxel_size_04x
        coords04x, inverse04x = torch.unique(coords04x, return_inverse=True, dim=0)
 
        coords08x = points[:, :3].clone()
        coords08x[:, 1:3] = (points_offsets - 1e-5) // self.voxel_size_08x
        coords08x, inverse08x = torch.unique(coords08x, return_inverse=True, dim=0)


        src = self.input_embed(points[:, 1:])

        src = src + self.pe0(pe_raw)
        src = self.mlp_vsa_layer_0(src, inverse01x, coords01x, self.grid_size)

        src = src + self.pe1(pe_raw)
        src = self.mlp_vsa_layer_1(src, inverse02x, coords02x, self.grid_size_02x)

        src = src + self.pe2(pe_raw)
        src = self.mlp_vsa_layer_2(src, inverse04x, coords04x, self.grid_size_04x)

        src = src + self.pe3(pe_raw)
        src = self.mlp_vsa_layer_3(src, inverse08x, coords08x, self.grid_size_08x)

        src = self.post_mlp(src)

        pillar_features = F.relu(torch_scatter.scatter_max(src, inverse01x, dim=0)[0])
        voxel_coords = coords01x[:, [0, 2, 1]]
        output = self.PointPillarScatter(pillar_features, voxel_coords) #batch dim w h
        
        return output