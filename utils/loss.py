import torch
import numpy as np
import torch.nn.functional as F

class YoloLoss(object):
    def __init__(self, 
                 device, 
                 num_classes,
                 anchor_size,
                 boxes_per_cell,
                 bbox_loss_weight,
                 objectness_loss_weight,
                 class_loss_weight
                 ):
         
        self.device = device
        self.num_classes = num_classes
        self.anchor_size = anchor_size
        self.boxes_per_cell = boxes_per_cell
        self.bbox_loss_weight = bbox_loss_weight
        self.objectness_loss_weight = objectness_loss_weight
        self.class_loss_weight = class_loss_weight
        self.anchor_boxes = np.array([[0., 0., anchor[0], anchor[1]] for anchor in anchor_size])  
    
    def compute_iou(self, anchor_boxes, gt_box):
        """
            anchor_boxes : ndarray -> [KA, 4] (cx, cy, bw, bh).
            gt_box : ndarray -> [1, 4] (cx, cy, bw, bh).
        """
        # anchors: [KA, 4]
        anchors = np.zeros_like(anchor_boxes)
        anchors[..., :2] = anchor_boxes[..., :2] - anchor_boxes[..., 2:] * 0.5  # x1y1
        anchors[..., 2:] = anchor_boxes[..., :2] + anchor_boxes[..., 2:] * 0.5  # x2y2
        anchors_area = anchor_boxes[..., 2] * anchor_boxes[..., 3]
        
        # gt_box: [1, 4] -> [KA, 4]
        gt_box = np.array(gt_box).reshape(-1, 4)
        gt_box = np.repeat(gt_box, anchors.shape[0], axis=0)
        gt_box_ = np.zeros_like(gt_box)
        gt_box_[..., :2] = gt_box[..., :2] - gt_box[..., 2:] * 0.5  # x1y1
        gt_box_[..., 2:] = gt_box[..., :2] + gt_box[..., 2:] * 0.5  # x2y2
        gt_box_area = np.prod(gt_box[..., 2:] - gt_box[..., :2], axis=1)

        # intersection
        inter_w = np.minimum(anchors[:, 2], gt_box_[:, 2]) - \
                  np.maximum(anchors[:, 0], gt_box_[:, 0])
        inter_h = np.minimum(anchors[:, 3], gt_box_[:, 3]) - \
                  np.maximum(anchors[:, 1], gt_box_[:, 1])
        inter_area = inter_w * inter_h
        
        # union
        union_area = anchors_area + gt_box_area - inter_area

        # iou
        iou = inter_area / union_area
        iou = np.clip(iou, a_min=1e-10, a_max=1.0)
        
        return iou
    
    def decode_gt(self, targets, strides, feature_size):
        batch_size = len(targets)
        gt_objes = [torch.zeros([batch_size, fmp_h, fmp_w, self.boxes_per_cell, 1])
            for (fmp_h, fmp_w) in feature_size
            ]
        gt_clses = [torch.zeros([batch_size, fmp_h, fmp_w, self.boxes_per_cell, self.num_classes]) 
            for (fmp_h, fmp_w) in feature_size
            ]
        gt_boxes =  [
            torch.zeros([batch_size, fmp_h, fmp_w, self.boxes_per_cell, 4])
            for (fmp_h, fmp_w) in feature_size
            ]

        for batch_index, target in enumerate(targets):
            target_cls = target['labels'].numpy()
            target_box = target['boxes'].numpy()

            for box, label in zip(target_box, target_cls):
                x1, y1, x2, y2 = box

                xc, yc = (x2 + x1) * 0.5, (y2 + y1) * 0.5
                bw, bh = x2 - x1, y2 - y1
                box = [0, 0, bw, bh]

                # check
                if bw < 1. or bh < 1.:
                    continue 

                # compute IoU
                iou = self.compute_iou(self.anchor_boxes, box)
                iou_mask = (iou > 0.5) # iou_threshold = 0.5

                label_assignment_results = []
                if iou_mask.sum() == 0:
                    iou_ind = np.argmax(iou)

                    level = iou_ind // 3             # pyramid level
                    anchor_idx = iou_ind - level * 3

                    stride = strides[level]

                    # compute the grid cell
                    xc_s = xc / stride
                    yc_s = yc / stride
                    grid_x = int(xc_s)
                    grid_y = int(yc_s)

                    label_assignment_results.append([grid_x, grid_y, level, anchor_idx])
                else:
                    for iou_ind, iou_m in enumerate(iou_mask):
                        if iou_m:
                            level = iou_ind // 3              # pyramid level
                            anchor_idx = iou_ind - level * 3  # anchor index

                            stride = strides[level]

                            xc_s = xc / stride
                            yc_s = yc / stride
                            grid_x = int(xc_s)
                            grid_y = int(yc_s)

                            label_assignment_results.append([grid_x, grid_y, level, anchor_idx])

                for result in label_assignment_results:
                    grid_x, grid_y, level, anchor_idx = result
                    fmp_h, fmp_w = feature_size[level]

                    if grid_x < fmp_w and grid_y < fmp_h:
                        gt_objes[level][batch_index, grid_y, grid_x, anchor_idx] = 1.0

                        cls_ont_hot = torch.zeros(self.num_classes)
                        cls_ont_hot[int(label)] = 1.0
                        gt_clses[level][batch_index, grid_y, grid_x, anchor_idx] = cls_ont_hot
                        
                        gt_boxes[level][batch_index, grid_y, grid_x, anchor_idx] = torch.tensor([x1, y1, x2, y2])

        gt_obj = torch.cat([gt.view(batch_size, -1, 1) for gt in gt_objes], dim=1)
        gt_box = torch.cat([gt.view(batch_size, -1, 4) for gt in gt_boxes], dim=1).float().to(self.device)
        gt_cls = torch.cat([gt.view(batch_size, -1, self.num_classes) for gt in gt_clses], dim=1).float().to(self.device)

        gt_obj = gt_obj.view(-1).float().to(self.device)               # [BM,]
        gt_cls = gt_cls.view(-1, self.num_classes).float().to(self.device)  # [BM, C]
        gt_box = gt_box.view(-1, 4).float().to(self.device) 

        return gt_obj, gt_cls, gt_box

    def loss_objectness(self, pred_obj, gt_obj):
        loss_obj = F.binary_cross_entropy_with_logits(pred_obj, gt_obj, reduction='none')

        return loss_obj
    
    def loss_classes(self, pred_cls, gt_cls):
        loss_cls = F.binary_cross_entropy_with_logits(pred_cls, gt_cls, reduction='none')

        return loss_cls

    def loss_bboxes(self, pred_box, gt_box):
        # regression loss
        gt_area = (gt_box[..., 2] - gt_box[..., 0]) * (gt_box[..., 3] - gt_box[..., 1])
        pred_area = (pred_box[..., 2] - pred_box[..., 0]).clamp_(min=0) * (pred_box[..., 3] - pred_box[..., 1]).clamp_(min=0)

        w_intersect = (torch.min(gt_box[..., 2], pred_box[..., 2]) - torch.max(gt_box[..., 0], pred_box[..., 0])).clamp_(min=0)
        h_intersect = (torch.min(gt_box[..., 3], pred_box[..., 3]) - torch.max(gt_box[..., 1], pred_box[..., 1])).clamp_(min=0)

        area_intersect = w_intersect * h_intersect
        area_union = gt_area + pred_area - area_intersect
        ious = area_intersect / area_union.clamp(min=torch.finfo(torch.float32).eps)

        ## GIoU
        g_w_intersect = torch.max(gt_box[..., 2], pred_box[..., 2]) - torch.min(gt_box[..., 0], pred_box[..., 0])
        g_h_intersect = torch.max(gt_box[..., 3], pred_box[..., 3]) - torch.min(gt_box[..., 1], pred_box[..., 1])
        ac_uion = g_w_intersect * g_h_intersect
        gious = ious - (ac_uion - area_union) / ac_uion.clamp(min=torch.finfo(torch.float32).eps)
        
        return 1-gious, ious

    def __call__(self, outputs, targets):
        gt_obj, gt_cls, gt_box = self.decode_gt(
                                                targets = targets,
                                                strides = outputs['stride'],
                                                feature_size = outputs['fmp_size']
                                                )

        # List[B, M, C] -> [B, M, C] -> [BM, C]
        pred_obj = torch.cat(outputs['pred_obj'], dim=1).view(-1)                     # [BM,]
        pred_cls = torch.cat(outputs['pred_cls'], dim=1).view(-1, self.num_classes)   # [BM, C]
        pred_box = torch.cat(outputs['pred_box'], dim=1).view(-1, 4)                  # [BM, 4]

        pos_masks = (gt_obj > 0)
        num_fgs = pos_masks.sum()

        # box loss
        pred_box_pos = pred_box[pos_masks]
        gt_bboxes_pos = gt_box[pos_masks]
        loss_box, ious = self.loss_bboxes(pred_box_pos, gt_bboxes_pos)
        loss_box = loss_box.sum() / num_fgs

        # cls loss
        pred_cls_pos = pred_cls[pos_masks]
        gt_classes_pos = gt_cls[pos_masks] * ious.unsqueeze(-1).clamp(0.)
        loss_cls = self.loss_classes(pred_cls_pos, gt_classes_pos)
        loss_cls = loss_cls.sum() / num_fgs
        
        # obj loss
        loss_obj = self.loss_objectness(pred_obj, gt_obj)
        loss_obj = loss_obj.sum() / num_fgs

        # total loss
        losses = self.objectness_loss_weight * loss_obj + \
                 self.class_loss_weight * loss_cls + \
                 self.bbox_loss_weight * loss_box
                 
        return dict(
                loss_obj = loss_obj,
                loss_cls = loss_cls,
                loss_box = loss_box,
                losses = losses
        )
    
def build_loss(args, device):
    loss =  YoloLoss(device = device,
                 anchor_size = args.anchor_size,
                 num_classes = args.num_classes,
                 boxes_per_cell = args.boxes_per_cell,
                 bbox_loss_weight = args.bbox_loss_weight,
                 objectness_loss_weight = args.objectness_loss_weight,
                 class_loss_weight = args.class_loss_weight)
    return loss
    