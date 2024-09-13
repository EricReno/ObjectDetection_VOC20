import torch

def refine_targets(self, targets, min_box_size):
    # rescale targets
    for tgt in targets:
        boxes = tgt["boxes"].clone()
        labels = tgt["labels"].clone()
        # refine tgt
        tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
        min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
        keep = (min_tgt_size >= min_box_size)

        tgt["boxes"] = boxes[keep]
        tgt["labels"] = labels[keep]
    
    return targets

def rescale_image_targets(images, targets, stride, min_box_size, multi_scale_range=[0.5, 1.5]):
    """
        Deployed for Multi scale trick.
    """
    if isinstance(stride, int):
        max_stride = stride
    elif isinstance(stride, list):
        max_stride = max(stride)

    # During training phase, the shape of input image is square.
    old_img_size = images.shape[-1]
    min_img_size = old_img_size * multi_scale_range[0]
    max_img_size = old_img_size * multi_scale_range[1]

    # Choose a new image size
    new_img_size = random.randrange(int(min_img_size), int(max_img_size + max_stride), int(max_stride))

    if new_img_size / old_img_size != 1:
        # interpolate
        images = torch.nn.functional.interpolate(
                            input=images, 
                            size=new_img_size, 
                            mode='bilinear', 
                            align_corners=False)
    # rescale targets
    for tgt in targets:
        boxes = tgt["boxes"].clone()
        labels = tgt["labels"].clone()
        boxes = torch.clamp(boxes, 0, old_img_size)
        # rescale box
        boxes[:, [0, 2]] = boxes[:, [0, 2]] / old_img_size * new_img_size
        boxes[:, [1, 3]] = boxes[:, [1, 3]] / old_img_size * new_img_size
        # refine tgt
        tgt_boxes_wh = boxes[..., 2:] - boxes[..., :2]
        min_tgt_size = torch.min(tgt_boxes_wh, dim=-1)[0]
        keep = (min_tgt_size >= min_box_size)

        tgt["boxes"] = boxes[keep]
        tgt["labels"] = labels[keep]

    return images, targets, new_img_size