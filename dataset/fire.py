import os
import cv2
import numpy as np
import torch.utils.data as data
import xml.etree.ElementTree as ET

# FIRE class names
# FIRE_CLASSES = ('fire', 'smoke')

class FIREDataset(data.Dataset):
    def __init__(self,
                 is_train :bool = False,
                 data_dir :str = None,
                 transform = None, 
                 image_set = 'train',
                 classes : list = []) -> None:
        super().__init__()

        self.is_train = is_train
        self.data_dir = data_dir
        self.transform = transform
        self.image_set = image_set
              
        self._imgpath = os.path.join('%s', 'JPEGImages', '%s.jpg')
        self._annopath = os.path.join('%s', 'Annotations', '%s.xml')
        self.class_to_ind = dict(zip(classes, range(len(classes))))

        self.ids = list()
        for line in open(os.path.join(self.data_dir, self.image_set+'.txt')):
            self.ids.append((self.data_dir, line.strip()))

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

        # guard against no boxes via resizing
        anno = np.array(anno).reshape(-1, 5)
        target = {
            "boxes": anno[:, :4],
            "labels": anno[:, 4],
            "orig_size": [image.shape[0], image.shape[1]]
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
                cur_pt = int(bbox.find(pt).text) - 1
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]+(0.1 if difficult else 0)
            bndbox.append(label_idx)
            anno += bndbox

        return np.array(anno).reshape(-1, 5), id