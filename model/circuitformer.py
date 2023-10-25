import torch
import torch.nn as nn

from model.voxelset.voxset import VoxSeT
import segmentation_models_pytorch as smp


def _break_up_pc(pc):
    for i in range(len(pc)):
        pc[i] = torch.unsqueeze(pc[i], dim=-1)
    x = (pc[0] + pc[2])/2
    y = (pc[1] + pc[3])/2
    width = pc[2] - pc[0]
    height = pc[3] - pc[1]
    area = width * height
    features = torch.concat([x, y, pc[0], pc[1], pc[2], pc[3], width, height, area],dim=-1)
    offset = pc[4]
    return features, offset



class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.PFN = VoxSeT()
    def forward(self, batch):
        points, offset = _break_up_pc(batch)
        points_list = []
        grid_size = 256
        for i in range(len(offset)):
            if i == 0:
                points_temp = points[:offset[i]]
                points_temp[:, 0] = points_temp[:, 0]/points_temp[:, 0].max() * grid_size
                points_temp[:, 1] = points_temp[:, 1]/points_temp[:, 1].max() * grid_size
                batch_index = i * torch.ones_like(points_temp[:, 0])
                points_list.append(torch.cat([batch_index[:,None],points_temp],dim=-1))
            else:
                points_temp = points[offset[i-1]:offset[i]]
                points_temp[:, 0] = points_temp[:, 0]/points_temp[:, 0].max() * grid_size
                points_temp[:, 1] = points_temp[:, 1]/points_temp[:, 1].max() * grid_size
                batch_index = i * torch.ones_like(points_temp[:, 0])
                points_list.append(torch.cat([batch_index[:,None],points_temp],dim=-1))
        feature = torch.cat(points_list,dim=0)
        return self.PFN(feature)





class CircuitFormer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()

        self.decoder = smp.UnetPlusPlus(encoder_name='resnet18', encoder_depth=5, encoder_weights=None , decoder_use_batchnorm=True, decoder_channels=(512, 256, 128, 64, 64), decoder_attention_type=None, in_channels=64, classes=1, activation='sigmoid', aux_params=None)
        ckpt = torch.load('../../../ckpts/resnet18.pth')
        ckpt.pop('conv1.weight')
        self.decoder.encoder.load_state_dict(ckpt, strict=False)

    def forward(self, batch):
        x = self.encoder(batch)
        output = self.decoder(x)
        return output