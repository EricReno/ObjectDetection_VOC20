import torch
import torch.nn as nn
try:
    from model.utils import Conv, CSPBlock
    from model.neck import SPPF
except:
    from utils import Conv, CSPBlock
    from neck import SPPF


## CSPDarkNet
class CSPDarkNet(nn.Module):
    def __init__(self, 
                 depth=1.0,
                 width=1.0, 
                 act_type = 'silu', 
                 norm_type='BN'):
        super(CSPDarkNet, self).__init__()
        self.feat_dims = [round(64 * width), 
                          round(128 * width), 
                          round(256 * width), 
                          round(512 * width), 
                          round(1024 * width)
                          ]
        
        # P1/2
        self.layer_1 = Conv(3, self.feat_dims[0], k=6, p=2, s=2, act_type=act_type, norm_type=norm_type)

        # P2/4
        self.layer_2 = nn.Sequential(
            Conv(self.feat_dims[0], self.feat_dims[1], k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(
                in_dim = self.feat_dims[1],
                out_dim= self.feat_dims[1],
                expand_ratio=0.5, 
                nblocks=round(3*depth), 
                shortcut=True, 
                act_type=act_type, 
                norm_type=norm_type)
        )

        # P3/8
        self.layer_3 = nn.Sequential(
            Conv(self.feat_dims[1], self.feat_dims[2], k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(
                in_dim = self.feat_dims[2],
                out_dim= self.feat_dims[2],
                expand_ratio=0.5, 
                nblocks=round(9*depth), 
                shortcut=True, 
                act_type=act_type, 
                norm_type=norm_type)
            )
        
        # P4/16
        self.layer_4 = nn.Sequential(
            Conv(self.feat_dims[2], self.feat_dims[3], k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(
                in_dim = self.feat_dims[3],
                out_dim= self.feat_dims[3],
                expand_ratio=0.5, 
                nblocks=round(9*depth), 
                shortcut=True, 
                act_type=act_type, 
                norm_type=norm_type)
            )
        
        # P4/32
        self.layer_5 = nn.Sequential(
            Conv(self.feat_dims[3], self.feat_dims[4], k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            SPPF(self.feat_dims[4], self.feat_dims[4], expand_ratio=0.5),
            CSPBlock(
                in_dim = self.feat_dims[4],
                out_dim= self.feat_dims[4],
                expand_ratio=0.5, 
                nblocks=round(3*depth), 
                shortcut=True, 
                act_type=act_type, 
                norm_type=norm_type)
            )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        return [c3, c4, c5]

# ImageNet-1K pretrained weight
model_urls = {
    "cspdarknet_n": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_nano.pth",
    "cspdarknet_t": None,  # For Medium-level, it is not necessary to load pretrained weight.
    "cspdarknet_s": "https://github.com/yjh0410/image_classification_pytorch/releases/download/weight/cspdarknet_small.pth",
    "cspdarknet_m": None,  # For Medium-level, it is not necessary to load pretrained weight.
    "cspdarknet_l": None,  # For Large-level,  it is not necessary to load pretrained weight.
    "cspdarknet_x": None,  # For Huge-level,   it is not necessary to load pretrained weight.
}

def build_backbone(model_name, pretrained):
    depth, width = 1.0, 1.0
    if model_name == 'cspdarknet_n':
        depth, width = 0.34, 0.25
    elif model_name == 'cspdarknet_t':
        depth, width = 0.34, 0.375
    elif model_name == 'cspdarknet_s':
        depth, width = 0.34, 0.50
    elif model_name == 'cspdarknet_m':
        depth, width = 0.67, 0.75
    elif model_name == 'cspdarknet_l':
        depth, width = 1.0, 1.0
    elif model_name == 'cspdarknet_x':
        depth, width = 1.34, 1.25
    else:
        print('Error: No {} backbone'.format(model_name))
        
    backbone = CSPDarkNet(
        depth=depth,
        width=width
    )
    feat_dims = backbone.feat_dims[-3:]
    
    if pretrained:
        url = model_urls[model_name]        
        if url is not None:
            print('Loading pretrained weight ...')
            checkpoint = torch.hub.load_state_dict_from_url(
                url=url, map_location="cpu", check_hash=True)
            # checkpoint state dict
            checkpoint_state_dict = checkpoint.pop("model")
            # model state dict
            model_state_dict = backbone.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    print('Unused key: ', k)

            backbone.load_state_dict(checkpoint_state_dict)
        else:
            print('No backbone pretrained: {}'.format(model_name)) 

    return backbone, feat_dims

if __name__ == "__main__":
    import time
    from thop import profile

    input = torch.randn(1, 3, 512, 512)

    model, _ = build_backbone(
        'cspdarknet_s',
        pretrained=True)
    
    t0 = time.time()
    outputs = model(input)
    t1 = time.time()
    print('Time:', t1-t0)
    for out in outputs:
        print(out.shape)

    flops, params = profile(model, inputs=(input, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))