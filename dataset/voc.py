import os
import cv2
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET

# VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
#                'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
#                'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

class VOCDataset(data.Dataset):
    def __init__(self,
                 is_train :bool = False,
                 data_dir :str = None, 
                 transform = None,
                 image_set :list = [],
                 voc_classes :list = []) -> None:
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

    def __getitem__(self, index):
        image, target = self.load_image_target(index)

        image, target, deltas = self.transform(image, target)
        
        return image, target, deltas
    
    def __len__(self):
        return len(self.ids)
    
    def __add__(self, other: data.Dataset) -> data.ConcatDataset:
        return super().__add__(other)
    
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

            label_idx = self.class_to_ind[name]*(-1 if difficult else 1)
            bndbox.append(label_idx)
            anno += bndbox

        return np.array(anno).reshape(-1, 5), id