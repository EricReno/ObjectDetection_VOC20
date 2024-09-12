import os
import cv2
import random
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET
from .augment.strong_augment import MixupAugment, MosaicAugment

# VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
#                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

class VOCDataset(data.Dataset):
    def __init__(self,
                 img_size,
                 is_train :bool = False,
                 data_dir :str = None, 
                 transform = None,
                 image_set :list = [],
                 voc_classes :list = [],
                 mosaic_augment :bool = False,
                 mixup_augment : bool = False) -> None:
        super().__init__()

        self.is_train = is_train
        self.data_dir = data_dir
        self.transform = transform
        self.image_set = image_set
        
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')

        self.class_to_ind = dict(zip(voc_classes, range(len(voc_classes))))

        self.ids = list()
        for (year, name) in self.image_set:
            rootpath = os.path.join(self.data_dir, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name+'.txt')):
                self.ids.append((rootpath, line.strip()))
        
        # 设置 mosaic 相关的参数
        self.mosaic_prob = 1.0 if mosaic_augment else 0.0
        self.mosaic_augment = MosaicAugment(img_size) if mosaic_augment else None

        # 设置 mixup 相关的参数
        self.mixup_prob = 0.15 if mixup_augment else 0.0
        self.mixup_augment = MixupAugment(img_size) if mixup_augment else None

        print('==============================')
        print('use Mosaic Augmentation: {}'.format(self.mosaic_prob))
        print('use Mixup Augmentation: {}'.format(self.mixup_prob))
    
    def __getitem__(self, index):
        if random.random() < self.mosaic_prob:
            # load a mosaic image
            mosaic = True
            image, target = self.load_mosaic(index)
        else:
            mosaic = False
            # load an image and target
            image, target = self.load_image_target(index)

        # MixUp
        if random.random() < self.mixup_prob:
            image, target = self.load_mixup(image, target)

        image, target, deltas = self.transform(image, target, mosaic=False)
        
        return image, target, deltas
    
    def __len__(self):
        return len(self.ids)
    
    def __add__(self, other: data.Dataset) -> data.ConcatDataset:
        return super().__add__(other)
    
    # ------------ Mosaic & Mixup ------------
    def load_mosaic(self, index):
        # ------------ Prepare 4 indexes of images ------------
        ## Load 4x mosaic image
        index_list = np.arange(index).tolist() + np.arange(index+1, len(self.ids)).tolist()
        id1 = index
        id2, id3, id4 = random.sample(index_list, 3)
        indexs = [id1, id2, id3, id4]

        ## Load images and targets
        image_list = []
        target_list = []
        for index in indexs:
            img_i, target_i = self.load_image_target(index)
            image_list.append(img_i)
            target_list.append(target_i)

        # ------------ Mosaic augmentation ------------
        image, target = self.mosaic_augment(image_list, target_list)

        return image, target
    
    def load_mixup(self, origin_image, origin_target):
        # ------------ Load a new image & target ------------
        new_index = np.random.randint(0, len(self.ids))
        new_image, new_target = self.load_mosaic(new_index)
            
        # ------------ Mixup augmentation ------------
        image, target = self.mixup_augment(origin_image, origin_target, new_image, new_target)

        return image, target
    
    def load_image_target(self, index):
        image, _ = self.pull_image(index)
        
        anno, _ = self.pull_anno(index)

        h, w = image.shape[:2]
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [h, w]
        }

        return image, target

    def pull_image(self, index):
        id = self.ids[index]
        image = cv2.imread(self._imgpath % id, cv2.IMREAD_COLOR)

        return image, id
    
    def pull_anno(self, index):
        id = self.ids[index]

        anno = []
        xml = ET.parse(self._annopath %id).getroot()
        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if self.is_train and difficult:
                continue

            bndbox = []
            bbox = obj.find('bndbox')
            name = obj.find('name').text.lower().strip()

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            for i, pt in enumerate(pts):
                cur_pt = float(bbox.find(pt).text)
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]+(0.1 if difficult else 0)
            bndbox.append(label_idx)
            anno += bndbox

        return np.array(anno).reshape(-1, 5), id