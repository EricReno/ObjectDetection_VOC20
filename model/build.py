import os
import torch
import torch.nn as nn
from .yolo import YOLO

def build_yolo(args, device, trainable):
    print('==============================')
    print('Build Model ...')
    print('Backbone: {}'.format(args.backbone))
    print('Neck    : {}'.format(args.neck.upper()))
    print('FPN     : {}'.format(args.fpn.upper()))
    print('Head    : Decoupled Head')
    print('')

    model = YOLO(device = device,
                trainable = trainable,
                backbone = args.backbone,
                neck = args.neck,
                fpn = args.fpn,
                anchor_size = args.anchor_size,
                num_classes = args.num_classes,
                nms_threshold = args.nms_threshold,
                boxes_per_cell = args.boxes_per_cell,
                confidence_threshold = args.confidence_threshold
                ).to(device)
    
    # -------------- Initialize YOLO --------------
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
    # Init bias
    init_prob = 0.01
    bias_value = -torch.log(torch.tensor((1. - init_prob) / init_prob))
    # obj pred
    for obj_pred in model.obj_preds:
        b = obj_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # cls pred
    for cls_pred in model.cls_preds:
        b = cls_pred.bias.view(1, -1)
        b.data.fill_(bias_value.item())
        cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
    # reg pred
    for reg_pred in model.reg_preds:
        b = reg_pred.bias.view(-1, )
        b.data.fill_(1.0)
        reg_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
        w = reg_pred.weight
        w.data.fill_(0.)
        reg_pred.weight = torch.nn.Parameter(w, requires_grad=True)

    # keep training
    if args.resume_weight_path and args.resume_weight_path != "None":
        ckpt_path = os.path.join('log', args.resume_weight_path)
        checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        # checkpoint state dict
        try:
            checkpoint_state_dict = checkpoint['model']
            print('Load model from the checkpoint: ', ckpt_path)
            model.load_state_dict(checkpoint_state_dict, strict=False)
            
            del checkpoint, checkpoint_state_dict
        except:
            print("No model in the given checkpoint.")

    return model