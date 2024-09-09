
import torch

def build_lambda_lr_scheduler(args, optimizer):
    epochs = args.epochs_total
    scheduler = args.lr_scheduler

    lrf = 0.01
    if scheduler == 'linear':
        lf = lambda x: (1 - x / epochs) * (1.0 - lrf) + lrf
    else:
        print('unknown lr scheduler.')
        exit(0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    return scheduler, lf