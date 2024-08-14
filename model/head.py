import torch.nn as nn
from model.utils import Conv

class Decouple(nn.Module):
    """
        This code referenced to https://github.com/ultralytics/yolov5
    """
    def __init__(self, input_channles):
        super(Decouple, self).__init__()

        self.cls_feats = nn.Sequential(
            Conv(c1=input_channles, c2=input_channles, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=input_channles, c2=input_channles, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
        )

        self.reg_feats = nn.Sequential(
            Conv(c1=input_channles, c2=input_channles, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
            Conv(c1=input_channles, c2=input_channles, k=3, p=1, s=1, act_type='silu', norm_type='BN'),
        )

    def forward(self, x):
        cls_feats = self.cls_feats(x)
        reg_feats = self.reg_feats(x)
       
        return cls_feats, reg_feats