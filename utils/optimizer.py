import os
import torch

def build_optimizer(args, model, resume=None):
    base_lr = args.learning_rate * args.batch_size * args.grad_accumulate / 64

    print('==============================')
    print('Optimizer: {}'.format(args.optimizer))
    print('--base lr: {}'.format(base_lr))
    print('--momentum: {}'.format(args.momentum))
    print('--weight_decay: {}'.format(args.weight_decay))

    # ------------- Divide model's parameters -------------
    param_dicts = [], [], []
    norm_names = ["norm"] + ["norm{}".format(i) for i in range(10000)]
    for n, p in model.named_parameters():
        if p.requires_grad:
            if "bias" == n.split(".")[-1]:
                param_dicts[0].append(p)      # no weight decay for all layers' bias
            else:
                if n.split(".")[-2] in norm_names:
                    param_dicts[1].append(p)  # no weight decay for all NormLayers' weight
                else:
                    param_dicts[2].append(p)  # weight decay for all Non-NormLayers' weight

    # Build optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(param_dicts[0], lr=base_lr, momentum=args.momentum, weight_decay=0.0)
    else:
        raise NotImplementedError("Unknown optimizer: {}".format(args.optimizer))
    
    # Add param groups
    optimizer.add_param_group({"params": param_dicts[1], "weight_decay": 0.0})
    optimizer.add_param_group({"params": param_dicts[2], "weight_decay": args.weight_decay})

    start_epoch = 0
    if args.resume_weight_path and args.resume_weight_path != 'None':
        ckpt_path = os.path.join('log', args.resume_weight_path)
        checkpoint = torch.load(ckpt_path, weights_only=False)
        # checkpoint state dict
        try:
            checkpoint_state_dict = checkpoint.pop("optimizer")
            print('Load optimizer from the checkpoint: ', args.resume_weight_path)
            optimizer.load_state_dict(checkpoint_state_dict)
            start_epoch = checkpoint.pop("epoch") + 1
            del checkpoint, checkpoint_state_dict
        except:
            print("No optimzier in the given checkpoint.")
    
    return optimizer, start_epoch