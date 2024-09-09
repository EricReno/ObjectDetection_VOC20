import torch
from thop import profile

def remove_thop_params(model):
    # 移除模型中的 total_ops 和 total_params
    for layer in model.modules():
        if hasattr(layer, 'total_ops'):
            del layer.total_ops
        if hasattr(layer, 'total_params'):
            del layer.total_params

def compute_flops(model, img_size, device):
    x = torch.randn(1, 3, img_size, img_size).to(device)
    print('')
    flops, params = profile(model, inputs=(x, ), verbose=False)
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))

    # 移除不必要的参数
    remove_thop_params(model)