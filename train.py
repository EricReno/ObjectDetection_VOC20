import os
import time
import torch
import numpy
from eval import Evaluator
from model.yolo import YOLO
from config import parse_args
from metric.criterion import Loss
from dataset.voc import VOCDataset
from dataset.utils import CollateFunc
from dataset.augment import Augmentation
from torch.utils.tensorboard import SummaryWriter

def train():
    parser, args = parse_args()
    writer = SummaryWriter('log')
    print("Setting Arguments.. : ")
    for action in parser._actions:
        if action.dest != 'help':
            print(f"{action.dest} = {getattr(args, action.dest)}")

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ---------------------------- Build Datasets ----------------------------
    val_transformer = Augmentation(is_train=False, image_size=args.image_size, transforms=args.data_augment)
    val_dataset = VOCDataset(is_train = False,
                             data_dir = args.data_root,
                             transform = val_transformer,
                             image_set = args.val_dataset,
                             voc_classes = args.class_names,
                             )
    val_sampler = torch.utils.data.RandomSampler(val_dataset)
    val_b_sampler = torch.utils.data.BatchSampler(val_sampler, args.batch_size, drop_last=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)
    
    train_transformer = Augmentation(is_train=True, image_size=args.image_size, transforms=args.data_augment)
    train_dataset = VOCDataset(is_train = False,
                               data_dir = args.data_root,
                               transform = train_transformer,
                               image_set = args.train_dataset,
                               voc_classes = args.class_names,
                               )
    
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_b_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    # ----------------------- Build Model ----------------------------------------
    model = YOLO(device = device,
                 trainable = True,
                 backbone = args.backbone,
                 anchor_size = args.anchor_size,
                 num_classes = args.num_classes,
                 nms_threshold = args.nms_threshold,
                 boxes_per_cell = args.boxes_per_cell,
                 confidence_threshold = args.confidence_threshold
                 ).to(device)
          
    criterion =  Loss(device = device,
                         anchor_size = args.anchor_size,
                         num_classes = args.num_classes,
                         boxes_per_cell = args.boxes_per_cell,
                         bbox_loss_weight = args.bbox_loss_weight,
                         objectness_loss_weight = args.objectness_loss_weight,
                         class_loss_weight = args.class_loss_weight)
    
    evaluator = Evaluator(
        device   =device,
        dataset  = val_dataset,
        ovthresh = args.nms_threshold,                        
        class_names = args.class_names,
        recall_thre = args.recall_threshold,
        visualization = args.eval_visualization)
    
    grad_accumulate = max(1, round(64 / args.batch_size))
    learning_rate = (grad_accumulate*args.batch_size/64)*args.learning_rate
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    max_mAP = 0
    start_epoch = 0
    # ----------------------- Build Train ----------------------------------------
    start = time.time()
    for epoch in range(start_epoch, args.epochs_total):
        model.train()
        train_loss = 0.0
        model.trainable = True
        for iteration, (images, targets) in enumerate(train_dataloader):
            ## learning rate
            ni = iteration + epoch * args.batch_size
            if epoch < args.warmup_epochs:
                optimizer.param_groups[0]['lr'] = numpy.interp(epoch*len(train_dataloader)+iteration,
                                                               [0, args.warmup_epochs*len(train_dataloader)],
                                                               [args.warmup_learning_rate, learning_rate])
            elif epoch >= args.second_stage_epochs:
                optimizer.param_groups[0]['lr'] = args.second_stage_lr
            elif epoch >= args.third_stage_epochs:
                optimizer.param_groups[0]['lr'] = args.third_stage_lr

            ## forward
            images = images.to(device)
            outputs = model(images)

            ## loss
            loss_dict = criterion(outputs=outputs, targets=targets)
            [loss_obj, loss_cls, loss_box, losses] = loss_dict.values()
            if grad_accumulate > 1:
               losses /= grad_accumulate
               loss_obj /= grad_accumulate
               loss_cls /= grad_accumulate
               loss_box /= grad_accumulate
            losses.backward()
            
            # optimizer.step
            if ni % grad_accumulate == 0:
                optimizer.step()
                optimizer.zero_grad()

            ## log
            print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f}, Loss_obj: {:8.4f}, Loss_cls: {:6.3f}, Loss_box: {:6.3f}".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epochs_total, iteration+1, len(train_dataloader), optimizer.param_groups[0]['lr'], losses, loss_obj, loss_cls, loss_box))
            train_loss += losses.item() * images.size(0)
        
        train_loss /= len(train_dataloader.dataset)
        writer.add_scalar('Loss/Train', train_loss, epoch)

        model.eval()
        val_loss = 0.0  
        with torch.no_grad():
            for iteration, (images, targets) in enumerate(val_dataloader):
                images = images.to(device)
                outputs = model(images)
                loss_dict = criterion(outputs=outputs, targets=targets)
                losses = loss_dict['losses'] #[loss_obj, loss_cls, loss_box, losses]
                print("Time [{}], Epoch [{}:{}/{}:{}], lr: {:.5f}, Loss: {:8.4f} ".format(time.strftime('%H:%M:%S', time.gmtime(time.time()- start)), 
                  epoch, args.epochs_total, iteration+1, len(val_dataloader), optimizer.param_groups[0]['lr'], losses))
                val_loss += losses.item() * images.size(0) 
            val_loss /= len(val_dataloader.dataset) 
            writer.add_scalar('Loss/val', val_loss, epoch)
        
        # save_model
        if epoch >= args.save_checkpoint_epoch:
            model.trainable = False
            model.nms_thresh = args.nms_threshold
            model.conf_thresh = args.confidence_threshold

            weight = '{}.pth'.format(epoch)
            ckpt_path = os.path.join(os.getcwd(), 'log', weight)
            if not os.path.exists(os.path.dirname(ckpt_path)): 
                os.makedirs(os.path.dirname(ckpt_path))
            
            with torch.no_grad():
                mAP = evaluator.eval(model)
            writer.add_scalar('mAP', mAP, epoch)
            print("Epoch [{}]".format('-'*100))
            print("Epoch [{}:{}], mAP [{:.4f}]".format(epoch, args.epochs_total, mAP))
            print("Epoch [{}]".format('-'*100))
            if mAP > max_mAP:
                torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'args': args},
                        ckpt_path)
                max_mAP = mAP

if __name__ == "__main__":
    train()