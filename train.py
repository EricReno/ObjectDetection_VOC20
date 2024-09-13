import os
import time
import torch
import numpy
from torch.utils.tensorboard import SummaryWriter

from config import parse_args
from evaluate import build_eval
from model.build import build_yolo
from utils.ema import ModelEMA
from utils.loss import build_loss
from utils.flops import compute_flops
from utils.optimizer import build_optimizer
from utils.lr_scheduler import build_lambda_lr_scheduler
from utils.rescale import refine_targets, rescale_image_targets
from dataset.build import build_transform, build_dataset, build_dataloader

def train():
    args = parse_args()
    writer = SummaryWriter('log')
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---------------------------- Build --------------------------
    val_transformer = build_transform(args, is_train=False)
    train_transformer = build_transform(args, is_train=True)

    val_dataset = build_dataset(args, is_train=False, transformer=val_transformer)
    train_dataset = build_dataset(args, True, train_transformer, )
    train_dataloader = build_dataloader(args, train_dataset)

    model = build_yolo(args, device, True)
    compute_flops(model, args.image_size, device)
          
    loss_function =  build_loss(args, device)
    
    evaluator = build_eval(args, val_dataset, device)
    
    optimizer, start_epoch = build_optimizer(args, model)

    lr_scheduler, lf = build_lambda_lr_scheduler(args, optimizer)
    if args.resume_weight_path and args.resume_weight_path != 'None':
        lr_scheduler.last_epoch = start_epoch - 1
        optimizer.step()
        lr_scheduler.step()
    
    if args.ema:
        model_ema = ModelEMA(model, start_epoch * len(train_dataloader))
    else:
        model_ema = None

    # ----------------------- Train --------------------------------
    print('==============================')
    max_mAP = 0
    start = time.time()
    for epoch in range(start_epoch, args.epochs_total+1):
        model.train()
        train_loss = 0.0
        model.trainable = True
        for iteration, (images, targets) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * len(train_dataloader)
            if epoch < args.warmup_epochs:
                for j, x in enumerate(optimizer.param_groups):
                    x['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration,
                                           [0, args.warmup_epochs*len(train_dataloader)],
                                           [0.1 if j == 0 else 0.0, x['initial_lr'] * lf(epoch)])
            
            images = images.to(device)
             # Multi scale
            if args.multi_scale:
                images, targets, img_size = rescale_image_targets(
                    images, targets, [8, 16, 32], args.min_box_size, args.multi_scale)
            else:
                targets = refine_targets(targets, args.min_box_size)

            ## forward
            outputs = model(images)

            ## loss
            loss_dict = loss_function(outputs=outputs, targets=targets)
            [loss_obj, loss_cls, loss_box, losses] = loss_dict.values()
            if args.grad_accumulate > 1:
               losses /= args.grad_accumulate
               loss_obj /= args.grad_accumulate
               loss_cls /= args.grad_accumulate
               loss_box /= args.grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % args.grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()
                # ema
                if model_ema is not None:
                   model_ema.update(model)

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f}, Loss_obj: {:8.4f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epochs_total, iteration+1, len(train_dataloader), optimizer.param_groups[2]['lr'], losses, loss_obj, loss_cls, loss_box))
            train_loss += losses.item() * images.size(0)

        lr_scheduler.step()

        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        # chech model
        model_eval = model if model_ema is None else model_ema.ema
        model_eval.eval()
        model_eval.trainable = False
        # save_model
        if epoch >= args.save_checkpoint_epoch:
            ckpt_path = os.path.join(os.getcwd(), 'log', '{}.pth'.format(epoch))
            if not os.path.exists(os.path.dirname(ckpt_path)):
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                mAP = evaluator.eval(model_eval)
            writer.add_scalar('mAP', mAP, epoch)

            if mAP > max_mAP:
                torch.save({
                        'model': model_eval.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'mAP':mAP,
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_mAP = mAP
        
if __name__ == "__main__":
    train()