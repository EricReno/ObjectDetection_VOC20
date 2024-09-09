import torch
import torch.nn as nn
from model.utils import Conv

# BottleNeck
class Bottleneck(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 expand_ratio=0.5,
                 shortcut=False,
                 depthwise=False,
                 act_type='silu',
                 norm_type='BN'):
        super(Bottleneck, self).__init__()
        inter_dim = int(out_dim * expand_ratio)  # hidden channels            
        self.cv1 = Conv(in_dim, inter_dim, k=1, norm_type=norm_type, act_type=act_type)
        self.cv2 = Conv(inter_dim, out_dim, k=3, p=1, norm_type=norm_type, act_type=act_type, depthwise=depthwise)
        self.shortcut = shortcut and in_dim == out_dim

    def forward(self, x):
        h = self.cv2(self.cv1(x))

        return x + h if self.shortcut else h

# ResBlock
class ResBlock(nn.Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 nblocks=1,
                 act_type='silu',
                 norm_type='BN'):
        super(ResBlock, self).__init__()
        assert in_dim == out_dim
        self.m = nn.Sequential(*[
            Bottleneck(in_dim, out_dim, expand_ratio=0.5, shortcut=True,
                       norm_type=norm_type, act_type=act_type)
                       for _ in range(nblocks)
                       ])

    def forward(self, x):
        return self.m(x)
    
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

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
    
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


model_urls = {
    "darknet_tiny": "https://github.com/EricReno/ImageClassification/releases/download/weight/darknet_tiny.pth",
    "darknet53": "https://github.com/EricReno/ImageClassification/releases/download/weight/darknet53.pth",
}

def build_backbone(model_name, pretrained):
    if model_name == 'darknet53':
        backbone = DarkNet53()
        feat_dims = backbone.feat_dims
    elif model_name == 'darknet_tiny':
        backbone = DarkNetTiny()
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
    model, _ = build_backbone(model_name='darknet53', pretrained=True)
    
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

