import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='You Only Look Once')

    parser.add_argument('--cuda', 
                        default=True,   
                        type=bool,
                        help='Enable CUDA for GPU acceleration.')   

    parser.add_argument('--num_workers',  
                        default=1,
                        type=int,
                        help='Number of CPU threads to use during data loading.')             
    
    # Dataset settings
    parser.add_argument('--data_root',
                        default='/data/VOCdevkit',
                        type=str,
                        help='Root directory of the dataset.')
    
    parser.add_argument('--train_dataset',
                        default=[('2007', 'trainval'), ('2012', 'trainval')],
                        help='Datasets used for training (format: [(year, split)]).')
    
    parser.add_argument('--val_dataset',
                        default=[('2007', 'test')],
                        help='Dataset split used for validation (format: [(year, split)]).')
    
    # Data augmentation
    parser.add_argument('--data_augment',
                        default= 'yolo',
                        choices = ['ssd', 'yolo'],
                        help='List of data augmentation techniques applied during training.')

    # Model settings
    parser.add_argument('--backbone', 
                        default='darknet_tiny',
                        type=str,
                        choices=['darknet_53', 'darknet_tiny',
                                 'cspdarknet_53', 'cspdarknet_tiny'],
                        help='Backbone network architecture.')
    parser.add_argument('--neck', 
                        default='sppf',
                        type=str,
                        choices=['sppf', 'csp_sppf'],
                        help='Neck network architecture.')
    parser.add_argument('--fpn', 
                        default='pafpn',
                        type=str,
                        choices=['fpn', 'pafpn'],
                        help='Feature Pyramid Network (FPN) architecture.')
    
    parser.add_argument('--image_size',
                        default=512,
                        type=int,
                        help='Input image size.')
    
    parser.add_argument('--num_classes',
                        default=20,
                        type=int,
                        help='Number of object classes.')
    
    parser.add_argument('--class_names',
                        default=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],          
                        type=str,
                        help='List of class names.')

    parser.add_argument('--anchor_size', 
                        default=[[10,13],[16,30],[33,23], # P3
                                 [30,61],[62,45],[59,119], # P4
                                 [116,90],[156,198],[373,326]],
                        help='Anchor box sizes (per feature map level).')
    
    parser.add_argument('--boxes_per_cell', 
                        default=3,
                        type=int,
                        help='Number of bounding boxes predicted per cell.')

    # Training settings
    parser.add_argument('--batch_size',
                        default=56,
                        type=int,
                        help='Batch size used during training (per GPU).')
    
    parser.add_argument('--epochs_total',
                        default=160,
                        type=int,
                        help='Total number of training epochs.')
    
    parser.add_argument('--warmup_epochs',
                        default=3,
                        type=int,
                        help='Number of warm-up epochs.')
    
    parser.add_argument('--save_checkpoint_epoch',
                        default=0,
                        type=int,
                        help='Epoch interval to save model checkpoints.')
    
    # Optimizer settings
    parser.add_argument('--optimizer',             
                        default='sgd',
                        type=str,
                        help='Base learning rate.')
    
    parser.add_argument('--lr_scheduler',             
                        default='linear',
                        type=str,
                        help='Base learning rate.')
    
    parser.add_argument('--grad_accumulate', 
                        default=1, type=int,
                        help='gradient accumulation')
    
    parser.add_argument('--learning_rate',             
                        default=0.01,
                        type=float,
                        help='Base learning rate.')
    
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum factor for SGD optimizer.')
    
    parser.add_argument('--weight_decay',
                        default=0.0005,
                        type=float,
                        help='Weight decay factor for regularization.')
    
     # Loss settings
    parser.add_argument('--bbox_loss_weight',
                        default=5.0,
                        type=float,
                        help='Weight for bounding box regression loss.')
    
    parser.add_argument('--objectness_loss_weight',
                        default=1.0,
                        type=float,
                        help='Weight for objectness loss.')
    
    parser.add_argument('--class_loss_weight',
                        default=1.0,
                        type=float,
                        help='Weight for classification loss.')
    
    # Thresholds
    parser.add_argument('--nms_threshold',
                        default=0.5,
                        type=float,
                        help='Threshold for non-maximum suppression (NMS).')
    
    parser.add_argument('--confidence_threshold',
                        default=0.3,
                        type=float,
                        help='Confidence threshold for filtering detections.')
    
    parser.add_argument('--recall_threshold',
                        default=101,
                        type=int,
                        help='Threshold for recall evaluation.')
    
    # Model checkpoint
    parser.add_argument('--model_weight_path',         
                        default='None',                
                        type=str,
                        help='Path to the initial model weights.')

    parser.add_argument('--resume_weight_path',         
                        default='None',                
                        type=str,
                        help='Path to the checkpoint from which to resume training.')
    
    parser.add_argument('--eval_visualization',         
                        default=False,                
                        type=bool,
                        help='Whether to visualize the evaluation results.')

    return parser.parse_args()