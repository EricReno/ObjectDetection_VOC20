import torch
import torch.nn as nn
try:
    from model.utils import Conv
except:
    from utils import Conv

class SPPF(nn.Module):
    def __init__(self, in_dim, out_dim, expand_ratio=0.5, pooling_size=5, act_type='silu', norm_type='BN'):
        super().__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(inter_dim * 4, out_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.MaxPool2d(kernel_size=pooling_size, stride=1, padding=pooling_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)

        return self.cv2(torch.cat((x, y1, y2, y3), 1))

# SPPF block with CSP module
class SPPFBlockCSP(nn.Module):
    """
        CSP Spatial Pyramid Pooling Block
    """
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 pooling_size=5,
                 act_type='lrelu',
                 norm_type='BN',
                 ):
        super(SPPFBlockCSP, self).__init__()
        inter_dim = int(in_dim * expand_ratio)
        self.out_dim = out_dim
        self.cv1 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.cv2 = Conv(in_dim, inter_dim, k=1, act_type=act_type, norm_type=norm_type)
        self.m = nn.Sequential(
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type),
            SPPF(inter_dim, 
                 inter_dim, 
                 expand_ratio=1.0, 
                 pooling_size=pooling_size, 
                 act_type=act_type, 
                 norm_type=norm_type),
            Conv(inter_dim, inter_dim, k=3, p=1, act_type=act_type, norm_type=norm_type)
        )
        self.cv3 = Conv(inter_dim * 2, self.out_dim, k=1, act_type=act_type, norm_type=norm_type)

        
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x)
        x3 = self.m(x2)
        y = self.cv3(torch.cat([x1, x3], dim=1))

        return y
     
def build_neck(neck_cfg, feat_dim):
    if neck_cfg == 'sppf':
        neck = SPPF(
            in_dim=feat_dim,
            out_dim=feat_dim,
            expand_ratio=0.5, 
            pooling_size=5,
            act_type='silu',
            norm_type='BN'
        )
    elif neck_cfg == 'csp_sppf':
        neck = SPPFBlockCSP(
            in_dim=feat_dim,
            out_dim=feat_dim,
            expand_ratio=0.5, 
            pooling_size=5,
            act_type='silu',
            norm_type='BN'
            )
    
    return neck