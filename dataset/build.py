import torch
from .voc import VOCDataset
from .utils import CollateFunc
from .augment import Augmentation

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
                          )
    return datasets
    
def build_augment(args, is_train):
    augment = Augmentation(
                           is_train=is_train, 
                           image_size=args.image_size, 
                           transforms=args.data_augment
                           )

    return augment

def build_dataloader(args, dataset):
    sampler = torch.utils.data.RandomSampler(dataset)
    b_sampler = torch.utils.data.BatchSampler(sampler, args.batch_size, drop_last=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=b_sampler, collate_fn=CollateFunc(), num_workers=args.num_workers, pin_memory=True)

    return dataloader