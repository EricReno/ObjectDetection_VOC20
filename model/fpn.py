import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils import Conv, ConvBlocks, CSPBlock

class FPN(nn.Module):
    def __init__(self, feat_dims):
        super(FPN, self).__init__()

        c3, c4, c5 = feat_dims
        out_dim = feat_dims[0]

        self.top_down_layer_1 = ConvBlocks(c5, int(0.5*c5))

        # P5 -> P4
        self.reduce_layer_1 = Conv(int(0.5*c5), int(0.25*c5), k=1)
        self.top_down_layer_2 = ConvBlocks(c4+int(0.25*c5), int(0.5*c4))

        # P4 -> P3
        self.reduce_layer_2 = Conv(int(0.5*c4), int(0.25*c4), k=1)
        self.top_down_layer_3 = ConvBlocks(c3+int(0.25*c4), int(0.5*c3))

        self.out_layers = nn.ModuleList([
                Conv(int(0.5*in_dim), out_dim, k=3) for in_dim in feat_dims])
        self.out_dim = [out_dim] * 3
    
    def forward(self, features):
        c3, c4, c5 = features
        
        # p5/32
        p5 = self.top_down_layer_1(c5)

        # p4/16
        p5_up = F.interpolate(self.reduce_layer_1(p5), scale_factor=2.0)
        p4 = self.top_down_layer_2(torch.cat([c4, p5_up], dim=1))
        
        # P3/8
        p4_up = F.interpolate(self.reduce_layer_2(p4), scale_factor=2.0)
        p3 = self.top_down_layer_3(torch.cat([c3, p4_up], dim=1))

        out_feats = [p3, p4, p5] # [P3, P4, P5]

        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
        return out_feats_proj
    
 
class PaFPN(nn.Module):
    def __init__(self, feat_dims, nblocks):
        super(PaFPN, self).__init__()

        c3, c4, c5 = feat_dims
        out_dim = feat_dims[0]

        # top down
        self.reduce_layer_1 = Conv(c5, int(0.5*c5), k=1)
        self.top_down_layer_1 = CSPBlock(in_dim = 2*c4,
                                         out_dim = c4,
                                         expand_ratio = 0.5,
                                         nblocks = nblocks,
                                         shortcut = False
                                         )
        
        self.reduce_layer_2 = Conv(c4, int(0.5*c4), k=1)
        self.top_down_layer_2 = CSPBlock(in_dim = 2*c3,
                                         out_dim = c3,
                                         expand_ratio = 0.5,
                                         nblocks = nblocks,
                                         shortcut = False
                                         )
        
        # bottom up
        self.reduce_layer_3 = Conv(c3, c3, k=3, p=1, s=2)
        self.bottom_up_layer_1 = CSPBlock(in_dim = 2*c3,
                                          out_dim = c4,
                                          expand_ratio = 0.5,
                                          nblocks = nblocks,
                                          shortcut = False
                                          )
        
        self.reduce_layer_4 = Conv(c4, c4, k=3, p=1, s=2)
        self.bottom_up_layer_2 = CSPBlock(in_dim = 2*c4,
                                          out_dim = c5,
                                          expand_ratio = 0.5,
                                          nblocks = nblocks,
                                          shortcut = False
                                          )

        self.out_layers = nn.ModuleList([
                Conv(in_dim, out_dim, k=1) for in_dim in feat_dims])
        self.out_dim = [out_dim] * 3
    
    def forward(self, features):
        c3, c4, c5 = features
        
        # top down
        c6 = self.reduce_layer_1(c5)
        c7 = F.interpolate(c6, scale_factor=2.0)
        c7 = torch.cat([c7, c4], dim=1)
        c7 = self.top_down_layer_1(c7)

        c7 = self.reduce_layer_2(c7)
        c8 = F.interpolate(c7, scale_factor=2.0)
        c8 = torch.cat([c8, c3], dim=1)
        c8_ = self.top_down_layer_2(c8)

        # bottom up
        c9 = self.reduce_layer_3(c8_)
        c9 = torch.cat([c9, c7], dim=1)
        c9_ = self.bottom_up_layer_1(c9)

        c10 = self.reduce_layer_4(c9_)
        c10 = torch.cat([c10, c6], dim=1)
        c10_ = self.bottom_up_layer_2(c10)

        out_feats = [c8_, c9_, c10_] # [P3, P4, P5]

        out_feats_proj = []
        for feat, layer in zip(out_feats, self.out_layers):
            out_feats_proj.append(layer(feat))
        return out_feats_proj

def build_fpn(backbone_cfg, fpn_cfg, feat_dims):
    if backbone_cfg in ['cspdarknet_tiny', 'darknet_tiny']:
        nblocks = 1
    elif backbone_cfg in ['cspdarknet_53', 'darknet_53']:
        nblocks = 3

    if fpn_cfg == 'fpn':
        fpn = FPN(feat_dims)
        feat_dims = fpn.out_dim
    elif fpn_cfg == 'pafpn':
        fpn = PaFPN(feat_dims, nblocks)
        feat_dims = fpn.out_dim

    return fpn, feat_dims