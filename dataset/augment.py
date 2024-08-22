import cv2
import torch
import numpy as np

class RandomHorizontalFlip(object):
    def __call__(self, image, boxes, labels):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1, :]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, labels

class RandomCenterCropPad(object):
    """
    Random center crop and random around padding for CornerNet.

    This operation generates randomly cropped image from the original image and
    pads it simultaneously. Different from :class:`RandomCrop`, the output
    shape may not equal to ``crop_size`` strictly. We choose a random value
    from ``ratios`` and the output shape could be larger or smaller than
    ``crop_size``. The padding operation is also different from :class:`Pad`,
    here we use around padding instead of right-bottom padding.

    The relation between output image (padding image) and original image:

    .. code:: text

                        output image

               +----------------------------+
               |          padded area       |
        +------|----------------------------|----------+
        |      |         cropped area       |          |
        |      |         +---------------+  |          |
        |      |         |    .   center |  |          | original image
        |      |         |        range  |  |          |
        |      |         +---------------+  |          |
        +------|----------------------------|----------+
               |          padded area       |
               +----------------------------+
    """
        
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            height, width, channel = image.shape
            for i in range(100):
                crop_width = np.random.randint(0.6*width, width)
                crop_height = np.random.randint(0.6*height, height)

                scale = np.random.choice((0.8, 1.0, 1.2))
                new_w = int(crop_width * scale)
                new_h = int(crop_height * scale)

                for j in range(50):
                    cropped_img = np.zeros((new_h, new_w, channel), dtype=image.dtype)
                    for c in range(channel):
                        cropped_img[:,:, c] += np.mean(image[::-1]).astype('uint8')

                    center_x = np.random.randint(0.3*width, 0.7*width)
                    center_y = np.random.randint(0.3*height, 0.7*height)

                    image_x0 = max(0, center_x - new_w//2)
                    image_x1 = min(width, center_x + new_w//2)
                    image_y0 = max(0, center_y - new_h//2)
                    image_y1 = min(height, center_y + new_h//2)

                    left, right = center_x - image_x0, image_x1 - center_x
                    top, bottom = center_y - image_y0, image_y1 - center_y

                    cropped_center_x = new_w // 2
                    cropped_center_y = new_h // 2

                    cropped_x0 = cropped_center_x - left
                    cropped_x1 = cropped_center_x + right
                    cropped_y0 = cropped_center_y - top
                    cropped_y1 = cropped_center_y + bottom

                    cropped_img[cropped_y0:cropped_y1,cropped_x0:cropped_x1,:] = image[image_y0:image_y1,image_x0:image_x1,:]
                    if len(boxes) == 0:
                        return cropped_img, boxes, labels

                    center = np.column_stack((boxes[:, 0] + boxes[:, 2] / 4, boxes[:, 1] + boxes[:, 3] / 4 ))  
                    mask = (center[:, 0] > image_x0) * (center[:, 1] > image_y0) * (center[:, 0] < image_x1) * (center[:, 1] < image_y1)
                    if not mask.any():
                        continue

                    filter_boxes = boxes[mask]
                    filter_boxes[:, 0] = np.clip(filter_boxes[:, 0] + (cropped_center_x - center_x), 0, None)
                    filter_boxes[:, 1] = np.clip(filter_boxes[:, 1] + (cropped_center_y - center_y), 0, None)
                    filter_boxes[:, 2] = np.clip(filter_boxes[:, 2] + (cropped_center_x - center_x), None, new_w)
                    filter_boxes[:, 3] = np.clip(filter_boxes[:, 3] + (cropped_center_y - center_y), None, new_h)

                    filter_labels = labels[mask]

                    return cropped_img, filter_boxes, filter_labels

            return image, boxes, labels
        
        return image, boxes, labels

class RandomSampleCrop(object):
    def __init__(self) -> None:
        self.sample_options = [
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ]

    def compute_ious(self, box_a, box_b):
        max_x_y = np.minimum(box_a[:, 2:], box_b[2:])
        min_x_y = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_x_y-min_x_y), a_min=0, a_max=np.inf)
        area_inter = inter[:, 0] * inter[:, 1]

        area_a = (box_a[:, 2]-box_a[:, 0]) * (box_a[:, 3]-box_a[:, 1])
        area_b = (box_b[2]-box_b[0]) * (box_b[3]-box_b[1])

        union = area_a + area_b - area_inter
        union = np.maximum(union, np.finfo(float).eps)
        
        return area_inter / union
    
    def __call__(self, image, boxes, labels):
        height, width, _ = image.shape
        while True:
            id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[id]

            if mode == None:
                return image, boxes, labels
            
            min_iou, max_iou = mode
            min_iou = min_iou if min_iou is not None else float('-inf')
            max_iou = max_iou if max_iou is not None else float('inf')

            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3*width, width)
                h = np.random.uniform(0.3*height, height)

                if h / w  < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(0, width - w)
                top = np.random.uniform(0, height - h)

                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                overlaps = self.compute_ious(boxes, rect)

                if overlaps.max() < min_iou or overlaps.min() > max_iou:
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # 判断中心点是否在裁剪框内
                mask = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1]) * \
                    (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()

                current_labels = labels[mask]

                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])

                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

class RandomExpand(object):
    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            height, width, depth = image.shape
            ratio = np.random.uniform(1, 4)
            left = np.random.uniform(0, width*ratio - width)
            top = np.random.uniform(0, height*ratio - height)

            expand_image = np.zeros(
                (int(height*ratio), int(width*ratio), depth),
                dtype=image.dtype)
            expand_image[int(top):int(top + height),
                        int(left):int(left + width)] = image
            image = expand_image

            boxes = boxes.copy()
            boxes[:, :2] += (int(left), int(top))
            boxes[:, 2:] += (int(left), int(top))

            return image, boxes, labels
        
        return image, boxes, labels
    
class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            delta = np.random.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0, 255)
        return image, boxes, labels

class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            alpha = np.random.uniform(self.lower, self.upper)
            image *= alpha
            image = np.clip(image, 0, 255)

        return image, boxes, labels

class RandomSaturationHue(object):
    def __init__(self, lower=0.5, upper=1.5, delta=18.0):
        self.lower = lower
        self.upper = upper
        self.delta = delta

        assert self.delta >= 0.0 and self.delta <= 360.0
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)

        if np.random.randint(2):
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0

        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        image = np.clip(image, 0, 255)

        return image, boxes, labels

class Augmentation():
    def __init__(self, 
                 is_train = True,
                 image_size = 608, 
                 transforms = None, 
                 ) -> None:

        self.is_train = is_train
        self.image_size = image_size

        self.transforms = []
        augments = {
            'RandomContrast'      : RandomContrast(),
            'RandomBrightness'    : RandomBrightness(),
            'RandomSaturationHue' : RandomSaturationHue(),

            'RandomExpand'        : RandomExpand(),
            'RandomSampleCrop'    : RandomSampleCrop(),
            'RandomHorizontalFlip': RandomHorizontalFlip(),
        }
        for t in transforms:
            self.transforms.append(augments[t])
    
    def __call__(self, image, target = None):   
        image = image.astype(np.float32).copy()
        boxes = target['boxes'].copy()
        labels = target['labels'].copy()

        if self.is_train:
            for t in self.transforms:
                image, boxes, labels = t(image, boxes, labels)

        # resize
        t_h, t_w = image.shape[:2]
        ratio = [self.image_size / t_w , self.image_size / t_h]

        image = cv2.resize(image, (self.image_size, self.image_size))
        if boxes is not None:
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * ratio[0]
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * ratio[1]

        ## to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        target['boxes'] = torch.from_numpy(boxes).float()
        target['labels'] = torch.from_numpy(labels).float()

        ## normalize image
        img_tensor /= 255.

        return img_tensor, target, ratio