import os
import cv2
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET

# VOC class names
VOC_CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')

class VOCDataset(data.Dataset):
    def __init__(self,
                 img_size :int = 640,
                 data_dir :str = None, 
                 image_sets = [('2007', 'trainval'), ('2012', 'trainval')],
                 transform = None,
                 is_train :bool = False) -> None:
        super().__init__()

        self.img_size = img_size
        self.image_set = image_sets
        self.is_train = is_train
        
        self.root = data_dir
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')

        self.ids = list()
        for (year, name) in self.image_set:
            rootpath = os.path.join(self.root, 'VOC' + year)
            for line in open(os.path.join(rootpath, 'ImageSets', 'Main', name+'.txt')):
                self.ids.append((rootpath, line.strip()))
        self.dataset_size = len(self.ids)

        self.class_to_ind = dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))

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
        image = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)

        return image, img_id
    
    def pull_anno(self, index):
        img_id = self.ids[index]

        anno = []
        xml = ET.parse(self._annopath %img_id).getroot()
        for obj in xml.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if difficult:
                continue

            name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            anno += bndbox

        return anno, img_id