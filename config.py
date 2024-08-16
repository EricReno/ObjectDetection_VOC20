import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='You Only Look Once')

    parser.add_argument('--cuda', 
                        default=True,                   
                        help='Whether to use CUDA for GPU acceleration.')
    
    parser.add_argument('--worker_number',  
                        default=6,
                        help='Number of logical processors.')

    parser.add_argument('--data_root',
                        default='/data/FaceDetection',
                        help='Root directory where the dataset is stored.')
    
    parser.add_argument('--datasets_val',
                        default='val',
                        help='Dataset split used for validation.')
    
    parser.add_argument('--datasets_train',
                        default='train', 
                        help='Datasets used for training.')
    
    parser.add_argument('--data_augment',
                        default=['RandomSaturationHue', 'RandomContrast', 'RandomBrightness', 'RandomExpand', 'RandomHorizontalFlip'],
                        # default=['RandomSaturationHue', 'RandomContrast', 'RandomBrightness', 'RandomSampleCrop', 'RandomExpand', 'RandomHorizontalFlip']
                        help='List of data augmentation techniques applied during training.')
    
    parser.add_argument('--backbone', 
                        default='darknet_tiny',
                        help="darknet53, darknet_tiny")

    parser.add_argument('--image_size',
                        default=608,
                        help='Input image size')
    
    parser.add_argument('--batch_size',
                        default=6,
                        help='Batch size used per GPU during training.')
    
    parser.add_argument('--class_names',
                        default= ['face'],          
                        help='List of classes the model predicts.')
    
    parser.add_argument('--classes_number',
                        default=1,
                        help='Number of classes.')
    

    parser.add_argument('--epoch_max',
                        default=160,
                        help='Maximum number of epochs.')
    
    parser.add_argument('--epoch_warmup',
                        default=1,
                        help='Number of epochs for warm-up.')
    
    parser.add_argument('--epoch_second',
                        default=60,
                        help='Epochs for the second phase of training.')
    
    parser.add_argument('--epoch_thirdly',
                        default=90,
                        help='Epochs for the final phase of training.')
    
    parser.add_argument('--epoch_save',
                        default=0,
                        help='Epoch to save model.')

    parser.add_argument('--lr',             
                        default=0.005,
                        help='Base learning rate for the model.')
    
    parser.add_argument('--lr_warmup',
                        default=0.00001,
                        help='Initial learning rate during model warm-up phase.')
    
    parser.add_argument('--lr_second',
                        default=0.0005,
                        help='Learning rate applied during the second phase of training.')
    
    parser.add_argument('--lr_thirdly',
                        default=0.00005,
                        help='Learning rate applied during the final phase of training.')
    
    parser.add_argument('--lr_momentum',
                        default=0.9,
                        help='Momentum for SGD.')
    
    parser.add_argument('--lr_weight_decay',
                        default=0.0005,
                        help='Weight decay for SGD.')

    parser.add_argument('--boxes_per_cell', 
                        default=3,
                        help='Number of bounding boxes predicted per cell.')
    
    parser.add_argument('--loss_box_weight',
                        default=5.0,
                        help='Weight for bounding box localization loss.')
    
    parser.add_argument('--loss_obj_weight',
                        default=1.0,
                        help='Weight for objectness loss.')
    
    parser.add_argument('--loss_cls_weight',
                        default=1.0,
                        help='Weight for class prediction loss.')

    parser.add_argument('--threshold_nms',
                        default=0.5,
                        help='Threshold for non-maximum suppression (NMS).')
    
    parser.add_argument('--threshold_conf',
                        default=0.2,
                        help='Confidence threshold for filtering detections.')
    
    parser.add_argument('--threshold_recall',
                        default=101,
                        help='Threshold for recall evaluation.')

    parser.add_argument('--anchor_size', default=[[10,13],[16,30],[33,23],     # P3
                                                  [30, 61],[62, 45],[59, 119],    # P4
                                                  [116,90],[156,198],[373, 326]], help='confidence threshold')
    
    parser.add_argument('--weight',         default='151.pth',                help='confidence threshold')


    return parser.parse_args()
