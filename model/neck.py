import torch
import torch.nn as nn
from model.utils import Conv
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, feat_dims):
        super(FPN, self).__init__()

        self.conv1 = nn.Sequential(
            Conv(c1=feat_dims[-3] + feat_dims[-2]//4, c2=feat_dims[-3]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-3]//2, c2=feat_dims[-3]//4, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-3]//4, c2=feat_dims[-3]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-3]//2, c2=feat_dims[-3]//4, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-3]//4, c2=feat_dims[-3]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
        )

            
        self.conv2 = nn.Sequential(
            Conv(c1=feat_dims[-2] + feat_dims[-1]//4, c2=feat_dims[-2]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-2]//2, c2=feat_dims[-2]//4, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-2]//4, c2=feat_dims[-2]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-2]//2, c2=feat_dims[-2]//4, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-2]//4, c2=feat_dims[-2]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
        )
        self.conv2_1 = Conv(c1=feat_dims[-2]//2, c2=feat_dims[-2]//4, k=1, p=0, s=1, act_type='silu', norm_type='BN')


        self.conv3 = nn.Sequential(
            Conv(c1=feat_dims[-1], c2=feat_dims[-1]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-1]//2, c2=feat_dims[-1]//4, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-1]//4, c2=feat_dims[-1]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-1]//2, c2=feat_dims[-1]//4, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=feat_dims[-1]//4, c2=feat_dims[-1]//2, k=1, p=0, s=1, act_type='silu', norm_type='BN'),
        )
        self.conv3_1 = Conv(c1=feat_dims[-1]//2, c2=feat_dims[-1]//4, k=1, p=0, s=1, act_type='silu', norm_type='BN')

    def forward(self, features):
        x1, x2, x3 = features

        f3 = self.conv3(x3)

        f3_up = F.interpolate(self.conv3_1(f3), scale_factor=2.0)
        f2 = self.conv2(torch.cat([f3_up, x2], dim=1))

        f2_up = F.interpolate(self.conv2_1(f2), scale_factor=2.0)
        f1 = self.conv1(torch.cat([f2_up, x1], dim=1))

        return [f1, f2, f3]