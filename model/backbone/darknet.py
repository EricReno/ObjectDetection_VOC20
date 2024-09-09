import torch
import torch.nn as nn
from model.utils import Conv, ResBlock, CSPBlock

## DarkNet-53
class DarkNet53(nn.Module):
    def __init__(self, ):
        super(DarkNet53, self).__init__()
        self.feat_dims = [256, 512, 1024]

        # P1
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1, act_type='silu', norm_type='BN'),
            Conv(32, 64, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(64, 64, nblocks=1, act_type='silu', norm_type='BN')
        )
        # P2
        self.layer_2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(128, 128, nblocks=2, act_type='silu', norm_type='BN')
        )
        # P3
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(256, 256, nblocks=8, act_type='silu', norm_type='BN')
        )
        # P4
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(512, 512, nblocks=8, act_type='silu', norm_type='BN')
        )
        # P5
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(1024, 1024, nblocks=4, act_type='silu', norm_type='BN')
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        return [c3, c4, c5]

## DarkNet-Tiny
class DarkNetTiny(nn.Module):
    def __init__(self,):
        super(DarkNetTiny, self).__init__()
        self.feat_dims = [64, 128, 256]

        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv(3, 16, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(16, 16, nblocks=1, act_type='silu', norm_type='BN')
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(16, 32, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(32, 32, nblocks=1, act_type='silu', norm_type='BN')
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(32, 64, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(64, 64, nblocks=3, act_type='silu', norm_type='BN')
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(128, 128, nblocks=3, act_type='silu', norm_type='BN')
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            ResBlock(256, 256, nblocks=2, act_type='silu', norm_type='BN')
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs

## CSPDarkNet-53
class CSPDarkNet53(nn.Module):
    def __init__(self, act_type = 'silu', norm_type='BN'):
        super(CSPDarkNet53, self).__init__()
        

        self.feat_dims = [256, 512, 1024]

        # P1
        self.layer_1 = nn.Sequential(
            Conv(3, 32, k=3, p=1, act_type='silu', norm_type='BN'),
            Conv(32, 64, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(64, 64, expand_ratio=0.5, nblocks=1, shortcut=True, act_type=act_type, norm_type=norm_type)
        )
        # P2
        self.layer_2 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(128, 128, expand_ratio=0.5, nblocks=2, shortcut=True, act_type=act_type, norm_type=norm_type)
        )

        # P3
        self.layer_3 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(256, 256, expand_ratio=0.5, nblocks=8, shortcut=True, act_type=act_type, norm_type=norm_type)
        )

        # P4
        self.layer_4 = nn.Sequential(
            Conv(256, 512, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(512, 512, expand_ratio=0.5, nblocks=8, shortcut=True, act_type=act_type, norm_type=norm_type)
        )
        # P5
        self.layer_5 = nn.Sequential(
            Conv(512, 1024, k=3, p=1, s=2, act_type='silu', norm_type='BN'),
            CSPBlock(1024, 1024, expand_ratio=0.5, nblocks=4, shortcut=True, act_type=act_type, norm_type=norm_type)
        )

    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        return [c3, c4, c5]

## CSPDarkNet-Tiny
class CSPDarkNetTiny(nn.Module):
    def __init__(self, act_type='silu', norm_type='BN'):
        super(CSPDarkNetTiny, self).__init__()
        self.feat_dims = [64, 128, 256]

        # stride = 2
        self.layer_1 = nn.Sequential(
            Conv(3, 16, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            CSPBlock(16, 16, expand_ratio=0.5, nblocks=1, shortcut=True, act_type=act_type, norm_type=norm_type)
        )
        # stride = 4
        self.layer_2 = nn.Sequential(
            Conv(16, 32, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            CSPBlock(32, 32, expand_ratio=0.5, nblocks=1, shortcut=True, act_type=act_type, norm_type=norm_type)
        )
        # stride = 8
        self.layer_3 = nn.Sequential(
            Conv(32, 64, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            CSPBlock(64, 64, expand_ratio=0.5, nblocks=3, shortcut=True, act_type=act_type, norm_type=norm_type)
        )
        # stride = 16
        self.layer_4 = nn.Sequential(
            Conv(64, 128, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            CSPBlock(128, 128, expand_ratio=0.5, nblocks=3, shortcut=True, act_type=act_type, norm_type=norm_type)
        )
        # stride = 32
        self.layer_5 = nn.Sequential(
            Conv(128, 256, k=3, p=1, s=2, act_type=act_type, norm_type=norm_type),
            CSPBlock(256, 256, expand_ratio=0.5, nblocks=2, shortcut=True, act_type=act_type, norm_type=norm_type)
        )


    def forward(self, x):
        c1 = self.layer_1(x)
        c2 = self.layer_2(c1)
        c3 = self.layer_3(c2)
        c4 = self.layer_4(c3)
        c5 = self.layer_5(c4)

        outputs = [c3, c4, c5]

        return outputs
    
model_urls = {
    "darknet_53": "https://github.com/EricReno/ImageClassification/releases/download/weight/darknet_53.pth",
    "darknet_tiny": "https://github.com/EricReno/ImageClassification/releases/download/weight/darknet_tiny.pth",
    "cspdarknet_53": "https://github.com/EricReno/ImageClassification/releases/download/weight/cspdarknet_53.pth",
    "cspdarknet_tiny": "https://github.com/EricReno/ImageClassification/releases/download/weight/cspdarknet_tiny.pth"
}

def build_backbone(model_name, pretrained):
    if model_name == 'darknet_53':
        backbone = DarkNet53()
        feat_dims = backbone.feat_dims
    elif model_name == 'darknet_tiny':
        backbone = DarkNetTiny()
        feat_dims = backbone.feat_dims
    elif model_name == 'cspdarknet53':
        backbone = CSPDarkNet53()
        feat_dims = backbone.feat_dims
    elif model_name == 'cspdarknet_tiny':
        backbone = CSPDarkNetTiny()
        feat_dims = backbone.feat_dims
    
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
            print('No backbone pretrained: DarkNet53') 

    return backbone, feat_dims


if __name__ == "__main__":
    import time
    from thop import profile

    input = torch.randn(1, 3, 608, 608)

    # darknet_tiny or darknet53
    model, _ = build_backbone(model_name='cspdarknet_tiny', pretrained=True)
    
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