import torch
from .voc import VOCDataset
from .utils import CollateFunc
from .augment.ssd_augment import SSDAugmentation
from .augment.yolo_augment import YOLOAugmentation

def build_dataset(args, is_train, transformer, image_set):
    if not is_train :
        print('==============================')
        print('Build Dataset: VOC ...')
        print('Dataset Class_names: {}'.format(args.class_names))

    datasets = VOCDataset(is_train = is_train,
                          data_dir = args.data_root,
                          transform = transformer,
                          image_set = image_set,
                          voc_classes = args.class_names,
                          mosaic_augment = False
                          )
    return datasets
    
def build_transform(args, is_train):
    if args.data_augment == 'ssd':
        transform = SSDAugmentation(is_train=is_train, image_size=args.image_size)
    elif args.data_augment == 'yolo':
        transform = YOLOAugmentation(is_train=is_train, image_size=args.image_size, max_stride=32)

    return transform

def build_dataloader(args, dataset):
    sampler = torch.utils.data.RandomSampler(dataset)
    b_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    return dataloader