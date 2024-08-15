import os
import cv2
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET

class FACEDataset(data.Dataset):
    def __init__(self,
                 img_size :int = 640,
                 data_dir :str = None, 
                 image_sets = 'val',
                 transform = None,
                 is_train :bool = False,
                 classnames :list = []) -> None:
        super().__init__()

        self.img_size = img_size
        self.image_set = image_sets
        self.is_train = is_train
        
        self.root = data_dir
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpeg')
        self._annopath = os.path.join('%s', 'Annotations', '%s.txt')

        self.ids = list()
        for line in open(os.path.join(self.root, self.image_set+'.txt')):
            self.ids.append((self.root, line.strip()))
        self.dataset_size = len(self.ids)

        self.class_to_ind = dict(zip(classnames, range(len(classnames))))

        self.transform = transform

    def __getitem__(self, index):
        image, target = self.load_image_target(index)

        image, target, deltas = self.transform(image, target, False)
        
        return image, target, deltas
    
    def __len__(self):
        return self.dataset_size
    
    def __add__(self, other: data.Dataset) -> data.ConcatDataset:
        return super().__add__(other)
    
    def load_image_target(self, index):

        image, _ = self.pull_image(index)
        
        anno, _ = self.pull_anno(index)

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [image.shape[0], image.shape[1]]
        }

        return image, target

    def pull_image(self, index):
        img_id = self.ids[index]
        label = self._annopath %img_id
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        return image, img_id
    
    def pull_anno(self, index):
        img_id = self.ids[index]
        label = self._annopath %img_id
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
    
        h, w = image.shape[:2]


        anno = []
        with open(label, 'r') as f:
            lines = f.readlines()
            for line in lines:
                bndbox = []
                line = line.strip().split()  # 去除首尾空格并按空格分割
                line = [float(num) for num in line]  # 将字符串转换为浮点数
                bndbox.append(line[1]*w - line[3]*w//2)
                bndbox.append(line[2]*w - line[4]*w//2)
                bndbox.append(line[1]*w + line[3]*w//2)
                bndbox.append(line[2]*w + line[4]*w//2)
                bndbox.append(line[0])

                anno += bndbox

        return anno, img_id