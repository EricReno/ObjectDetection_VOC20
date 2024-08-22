import torch
import numpy as np
import torch.nn as nn

import os
import sys
cur = os.path.dirname(os.path.abspath(__file__))
pro_path = os.path.abspath(os.path.join(cur, '..'))
sys.path.append(pro_path)

from config import parse_args
from model.head import Decouple
from model.neck import build_neck
from model.backbone.darknet import build_backbone

class YOLO(nn.Module):
    def __init__(self, 
                 device,
                 trainable,
                 backbone,
                 anchor_size,
                 num_classes,
                 nms_threshold,
                 boxes_per_cell,
                 confidence_threshold
                 ):
        super(YOLO, self).__init__()

        self.deploy = False
        self.device = device
        self.stride = [8, 16, 32]                            
        self.trainable = trainable
        self.num_classes = num_classes
        self.nms_threshold = nms_threshold
        self.boxes_per_cell = boxes_per_cell
        self.confidence_threshold = confidence_threshold
        self.anchor_size = torch.as_tensor(anchor_size).float().view(3, -1, 2) # [A, 2]   # 416 scale

        self.backbone, feat_dims = build_backbone(backbone, pretrained=trainable)
        
        self.neck, feat_dims = build_neck(feat_dims)

        self.heads = nn.ModuleList(
            [Decouple(dim) for dim in feat_dims]
        )
        
        self.obj_pred = nn.ModuleList(
            [nn.Conv2d(dim, 1*self.boxes_per_cell, kernel_size=1) for dim in feat_dims]
        )
        
        self.cls_pred = nn.ModuleList(
            [nn.Conv2d(dim, self.num_classes*self.boxes_per_cell, kernel_size=1) for dim in feat_dims]
        )
        
        self.reg_pred = nn.ModuleList(
            [nn.Conv2d(dim, 4*self.boxes_per_cell, kernel_size=1) for dim in feat_dims]
        )

    def create_grid(self, fmp_size):
        """ 
            用于生成G矩阵，其中每个元素都是特征图上的像素坐标。
        """
        # 特征图的宽和高
        ws, hs = fmp_size

        # 生成网格的x坐标和y坐标
        grid_y, grid_x = torch.meshgrid([torch.arange(hs), torch.arange(ws)],  indexing='ij')

        # 将xy两部分的坐标拼起来：[H, W, 2]
        grid_xy = torch.stack([grid_x, grid_y], dim=-1).float()

        # [H, W, 2] -> [H, W, 2, 2] -> [HW, 2] -> [HW, 2]
        grid_xy = torch.unsqueeze(grid_xy, dim=2)
        grid_xy = grid_xy.repeat(1, 1, self.boxes_per_cell, 1).view(-1, 2).to(self.device)

        return grid_xy

    def generate_anchors(self, level, fmp_size):
        """
            fmp_size: (List) [H, W]
        """
        fmp_h, fmp_w = fmp_size

        anchor_size = self.anchor_size[level]

        # generate grid cells
        anchor_y, anchor_x = torch.meshgrid([torch.arange(fmp_h), torch.arange(fmp_w)], indexing='ij')
        anchor_xy = torch.stack([anchor_x, anchor_y], dim=-1).float().view(-1, 2)

        # [HW, 2] -> [HW, A, 2] -> [M, 2]
        anchor_xy = anchor_xy.unsqueeze(1).repeat(1, self.boxes_per_cell, 1)
        anchor_xy = anchor_xy.view(-1, 2).to(self.device)

        # [A, 2] -> [1, A, 2] -> [HW, A, 2] -> [M, 2]
        anchor_wh = anchor_size.unsqueeze(0).repeat(fmp_h*fmp_w, 1, 1)
        anchor_wh = anchor_wh.view(-1, 2).to(self.device)

        anchors = torch.cat([anchor_xy, anchor_wh], dim=-1)

        return anchors
    
    def decode_boxes(self, anchors, reg_pred, level):
        """
            将txtytwth转换为常用的x1y1x2y2形式。
        """
        # 计算预测边界框的中心点坐标和宽高
        pred_ctr = (torch.sigmoid(reg_pred[..., :2]) + anchors[..., :2]) * self.stride[level]
        pred_wh = torch.exp(reg_pred[..., 2:]) * anchors[..., 2:]

        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = pred_ctr - pred_wh * 0.5
        pred_x2y2 = pred_ctr + pred_wh * 0.5
        pred_box = torch.cat([pred_x1y1, pred_x2y2], dim=-1)

        return pred_box

    ## basic NMS
    def nms(self, bboxes, scores, nms_thresh):
        """"Pure Python NMS."""
        # 将所有bbox的中心带你坐标和宽高换算成x1y1x2y2形式
        pred_x1y1 = bboxes[..., :2] - bboxes[..., 2:] * 0.5
        pred_x2y2 = bboxes[..., :2] + bboxes[..., 2:] * 0.5
        bboxes = np.concatenate([pred_x1y1, pred_x2y2], axis=-1)

        x1 = bboxes[:, 0]  #xmin
        y1 = bboxes[:, 1]  #ymin
        x2 = bboxes[:, 2]  #xmax
        y2 = bboxes[:, 3]  #ymax

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            # compute iou
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(1e-10, xx2 - xx1)
            h = np.maximum(1e-10, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(iou <= nms_thresh)[0]
            order = order[inds + 1]

        return keep
  
    def postprocess(self, obj_preds, cls_preds, box_preds):
        """
        Input:
            obj_preds: List[np.array] -> [[M, 1], ...] or None
            cls_preds: List[np.array] -> [[M, C], ...]
            box_preds: List[np.array] -> [[M, 4], ...]
        Output:
            bboxes: [N, 4]
            scores: [N,]
            labels: [N,]
        """
        all_scores = []
        all_labels = []
        all_bboxes = []

        for obj_pred_i, cls_pred_i, box_pred_i in zip(obj_preds, cls_preds, box_preds):
            # [M, C] -> [MC,]
            scores_i = (torch.sqrt(obj_pred_i.sigmoid() * cls_pred_i.sigmoid())).flatten()

            topk_scores, topk_idxs = scores_i.sort(descending=True)

            keep_idxs = topk_scores > self.confidence_threshold
            scores = topk_scores[keep_idxs]
            topk_idxs = topk_idxs[keep_idxs]

            anchor_idxs = torch.div(topk_idxs, self.num_classes, rounding_mode='floor')
            labels = topk_idxs % self.num_classes

            bboxes = box_pred_i[anchor_idxs]

            all_scores.append(scores)
            all_labels.append(labels)
            all_bboxes.append(bboxes)
        
        scores = torch.cat(all_scores)
        labels = torch.cat(all_labels)
        bboxes = torch.cat(all_bboxes)

         # to cpu & numpy
        scores = scores.cpu().numpy()
        labels = labels.cpu().numpy()
        bboxes = bboxes.cpu().numpy()

        # nms
        keep = np.zeros(len(bboxes), dtype=np.int32)
        for i in range(self.num_classes):
            inds = np.where(labels == i)[0]
            if len(inds) == 0:
                continue
            c_bboxes = bboxes[inds]
            c_scores = scores[inds]
            c_keep = self.nms(c_bboxes, c_scores, self.nms_threshold)
            keep[inds[c_keep]] = 1
        keep = np.where(keep > 0)
        scores = scores[keep]
        labels = labels[keep]
        bboxes = bboxes[keep]
        
        return bboxes, scores, labels

    @torch.no_grad()
    def inference(self, x):
        backbone_feats = self.backbone(x) # [1, 3, 416, 416] --> [1, 256, 52, 52], [1, 512, 26, 26], [1, 1024, 13, 13]

        pyramid_feats = self.neck(backbone_feats) # --> [torch.Size([1, 128, 52, 52]), torch.Size([1, 256, 26, 26]), torch.Size([1, 512, 13, 13])]
        
        all_anchors = []
        all_obj_preds = []
        all_cls_preds = []
        all_box_preds = []

        for level, (feat, head) in enumerate(zip(pyramid_feats, self.heads)):
            cls_feat, reg_feat = head(feat)

            obj_pred = self.obj_pred[level](reg_feat)
            cls_pred = self.cls_pred[level](cls_feat)
            reg_pred = self.reg_pred[level](reg_feat)

            # anchors: [M, 2]
            fmp_size = obj_pred.shape[-2:]
            anchors = self.generate_anchors(level, fmp_size)

            # 对 pred 的size做一些view调整，便于后续的处理
            # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
            obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(-1, 1)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes)
            reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4)

            # 解算边界框, 并归一化边界框: [H*W, 4]
            box_pred = self.decode_boxes(anchors, reg_pred, level)

            all_obj_preds.append(obj_pred)
            all_cls_preds.append(cls_pred)
            all_box_preds.append(box_pred)
            all_anchors.append(anchors)

        if self.deploy:
            obj_preds = torch.cat(all_obj_preds, dim=0)
            cls_preds = torch.cat(all_cls_preds, dim=0)
            box_preds = torch.cat(all_box_preds, dim=0)
            scores = torch.sqrt(obj_preds.sigmoid() * cls_preds.sigmoid())
            bboxes = box_preds
            # [n_anchors_all, 4 + C]
            outputs = torch.cat([bboxes, scores], dim=-1)
        else:
            bboxes, scores, labels = self.postprocess(
                all_obj_preds, all_cls_preds, all_box_preds)

            outputs = {
                    "scores": scores,
                    "labels": labels,
                    "bboxes": bboxes
            }

        return outputs

    def forward(self, x):
        if not self.trainable:
            return self.inference(x)
        else:
            batch_size = x.shape[0]

            # [1, 3, 416, 416] --> [[1, 256, 52, 52], [1, 512, 26, 26], [1, 1024, 13, 13]]
            backbone_feats = self.backbone(x) 
            
            # --> [torch.Size([1, 128, 52, 52]), torch.Size([1, 256, 26, 26]), torch.Size([1, 512, 13, 13])]
            pyramid_feats = self.neck(backbone_feats) 

            all_fmp_sizes = []
            all_obj_preds = []
            all_cls_preds = []
            all_box_preds = []
            for level, (feat, head) in enumerate(zip(pyramid_feats, self.heads)):
                cls_feat, reg_feat = head(feat)

                obj_pred = self.obj_pred[level](reg_feat)
                cls_pred = self.cls_pred[level](cls_feat)
                reg_pred = self.reg_pred[level](reg_feat)

                fmp_size = obj_pred.shape[-2:]

                # anchors: [M, 2]
                anchors = self.generate_anchors(level, fmp_size)

                # 对 pred 的size做一些view调整，便于后续的处理
                # [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
                obj_pred = obj_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 1)
                cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.num_classes)
                reg_pred = reg_pred.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

                box_pred = self.decode_boxes(anchors, reg_pred, level)
                
                all_obj_preds.append(obj_pred)
                all_cls_preds.append(cls_pred)
                all_box_preds.append(box_pred)
                all_fmp_sizes.append(fmp_size)

        outputs = {"pred_obj": all_obj_preds,                 # (List) [B, M, 1]
                   "pred_cls": all_cls_preds,                 # (List) [B, M, C]
                   "pred_box": all_box_preds,                 # (List) [B, M, 4]
                   "stride": self.stride,                     # (Int)
                   "fmp_size": all_fmp_sizes                  # (List) [fmp_h, fmp_w]
                   }
        
        return outputs

if __name__ == "__main__":
    import time
    from thop import profile
    parser, args = parse_args()

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    input = torch.randn(1, 3, args.image_size, args.image_size).to(device)

    model = YOLO(device = device,
                 trainable= True,
                 backbone = args.backbone,
                 anchor_size = args.anchor_size,
                 num_classes = args.num_classes,
                 nms_threshold = args.nms_threshold,
                 boxes_per_cell = args.boxes_per_cell,
                 confidence_threshold = args.confidence_threshold
                 ).to(device)

    outputs = model(input)
    print(model)
    
    flops, params = profile(model, inputs=(input, ), verbose=False)
    print('==============================')
    print('GFLOPs : {:.2f}'.format(flops / 1e9 * 2))
    print('Params : {:.2f} M'.format(params / 1e6))