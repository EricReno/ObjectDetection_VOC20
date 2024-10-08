
import cv2
import random
import numpy as np
try:
    from dataset.augment.yolo_augment import random_perspective
except:
    from yolo_augment import random_perspective

# ------------------------- Strong augmentations -------------------------
## Mosaic Augmentation
class MosaicAugment(object):
    def __init__(self, 
                 img_size) -> None:
        self.img_size = img_size

    def __call__(self, image_list, target_list):
        assert len(image_list) == 4

        mosaic_img = np.ones([self.img_size*2, self.img_size*2, image_list[0].shape[2]], dtype=np.uint8) * 114
        
        # mosaic center
        yc, xc = [int(random.uniform(-x, 2*self.img_size + x)) for x in [-self.img_size // 2, -self.img_size // 2]]

        mosaic_bboxes = []
        mosaic_labels = []
        for i in range(4):
            img_i, target_i = image_list[i], target_list[i]

            bboxes_i = target_i["boxes"]
            labels_i = target_i["labels"]

            orig_h, orig_w, _ = img_i.shape

            # resize
            r = self.img_size / max(orig_h, orig_w)
            if r != 1: 
                interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                img_i = cv2.resize(img_i, (int(orig_w * r), int(orig_h * r)), interpolation=interp)
            h, w, _ = img_i.shape

            # place img in img4
            if i == 0:  # top left
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w-(x2a-x1a), h-(y2a-y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2*self.img_size), yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            
            mosaic_img[y1a:y2a, x1a:x2a] = img_i[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            # labels
            bboxes_i_ = bboxes_i.copy()
            if len(bboxes_i) > 0:
                # a valid target, and modify it.
                bboxes_i_[:, 0] = (w * bboxes_i[:, 0] / orig_w + padw)
                bboxes_i_[:, 1] = (h * bboxes_i[:, 1] / orig_h + padh)
                bboxes_i_[:, 2] = (w * bboxes_i[:, 2] / orig_w + padw)
                bboxes_i_[:, 3] = (h * bboxes_i[:, 3] / orig_h + padh)  

                mosaic_bboxes.append(bboxes_i_)
                mosaic_labels.append(labels_i)
        
        if len(mosaic_bboxes) == 0:
            mosaic_bboxes = np.array([]).reshape(-1, 4)
            mosaic_labels = np.array([]).reshape(-1)
        else:
            mosaic_bboxes = np.concatenate(mosaic_bboxes)
            mosaic_labels = np.concatenate(mosaic_labels)

        # clip
        mosaic_bboxes = mosaic_bboxes.clip(0, self.img_size * 2)

        # random perspective
        mosaic_targets = np.concatenate([mosaic_labels[..., None], mosaic_bboxes], axis=-1)
        mosaic_img, mosaic_targets = random_perspective(
            mosaic_img,
            mosaic_targets,
            degrees     = 0.0,
            translate   = 0.1,
            scale       = [0.5, 1.5],
            shear       = 0.0,
            perspective = 0.0,
            border=[-self.img_size//2, -self.img_size//2]
            )
        
        # target
        mosaic_target = {
            "boxes": mosaic_targets[..., 1:],
            "labels": mosaic_targets[..., 0],
            "orig_size": [self.img_size, self.img_size]
        }

        return mosaic_img, mosaic_target
    
class MixupAugment(object):
    def __init__(self,
                 img_size) -> None:
        self.img_size = img_size

    def __call__(self, origin_image, origin_target, new_image, new_target):
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        mixup_image = r * origin_image.astype(np.float32) + \
                    (1.0 - r)* new_image.astype(np.float32)
        mixup_image = mixup_image.astype(np.uint8)

        cls_labels = new_target["labels"].copy()
        box_labels = new_target["boxes"].copy()

        mixup_bboxes = np.concatenate([origin_target["boxes"], box_labels], axis=0)
        mixup_labels = np.concatenate([origin_target["labels"], cls_labels], axis=0)

        mixup_target = {
            "boxes": mixup_bboxes,
            "labels": mixup_labels,
            'orig_size': mixup_image.shape[:2]
        }
        
        return mixup_image, mixup_target
        