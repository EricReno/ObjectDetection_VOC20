import os
import cv2
import torch
import numpy as np
from typing import List
from config import parse_args
from model.yolov3 import YOLOv3

import xml.etree.ElementTree as ET
from dataset.face import FACEDataset
from dataset.augment import Augmentation

def rescale_bboxes(bboxes, origin_size, ratio):
    # rescale bboxes
    if isinstance(ratio, float):
        bboxes /= ratio
    elif isinstance(ratio, List):
        bboxes[..., [0, 2]] /= ratio[0]
        bboxes[..., [1, 3]] /= ratio[1]
        origin_size = [
            origin_size[0]/ratio[0],
            origin_size[1]/ratio[1]
            ]
    else:
        raise NotImplementedError("ratio should be a int or List[int, int] type.")

    # clip bboxes
    bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_size[0])
    bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_size[1])

    return bboxes

class Evaluator():
    """ VOC AP Evaluation class"""
    def __init__(self,
                 device,
                 data_dir,
                 dataset,
                 image_sets,
                 ovthresh,
                 class_names,
                 recall_thre) -> None:
        
        self.device = device
        self.data_dir = data_dir
        self.dataset = dataset
        self.image_sets = image_sets
        self.ovthresh = ovthresh
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.recall_thre = recall_thre
        self.num_images = len(self.dataset)
        # all_boxes[cls][image] = N x 5 array of detections in (x1, y1, x2, y2, score)
        self.all_boxes = [[[] for _ in range(self.num_images)
                          ] for _ in range(self.num_classes)
                         ]

    def parse_rec(self, filename):
        image_path = filename.replace('Annotations', 'JPEGImages').replace('.txt', '.jpeg')
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        h, w = image.shape[:2]

        objects = []
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                obj_struct = {}
                line = line.strip().split()  # 去除首尾空格并按空格分割
                line = [float(num) for num in line]  # 将字符串转换为浮点数

                obj_struct['name'] = 'face'
                obj_struct['difficult'] = 0
                obj_struct['bbox'] = [int(line[1]*w - line[3]*w//2),
                                    int(line[2]*h - line[4]*h//2),
                                    int(line[1]*w + line[3]*w//2),
                                    int(line[2]*h + line[4]*h//2)]
            objects.append(obj_struct)

        return objects

    def voc_ap(self, recall, precision):
        """ 
        Compute the average precision, given the recall and precision curves.
        Code originally from https://github.com/rbgirshick/py-faster-rcnn.
        
        # Arguments
            recall:    The recall curve (np.array).
            precision: The precision curve (np.array).
        # Returns
            The average precision as computed in py-faster-rcnn.
        """
        
        # first append sentinel values at the end
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([1.0], precision, [0.0]))
        
        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
            
        # where X axis (recall) changes value， #excalidraw ,np.where()返回下标索引数组组成的元组
        i = np.where(mrec[:-1] != mrec[1:])[0]     

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
  
    def inference(self, model):
        for i in range(self.num_images):
            img, target, deltas = self.dataset.__getitem__(i)
            orig_h, orig_w = img.shape[1:]

            # preprocess
            img = img.unsqueeze(0).to(self.device)

            # forward
            import time
            start = time.time()
            outputs = model(img)
            end = time.time()

            print('infer:', end-start)

            scores = outputs['scores']
            labels = outputs['labels']
            bboxes = outputs['bboxes']

            # rescale bboxes
            bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], deltas)

            # if len(bboxes) == 0:
            #     continue
            # else:
            #     show_img, _ = self.dataset.pull_image(i)
            #     for box in bboxes:
            #         cv2.rectangle(show_img, (int(box[0]), int(box[1])),  (int(box[2]), int(box[3])), (255, 0, 0))


            #     print(self.class_names[labels[0]], ":", scores)
            #     cv2.imshow('1', show_img)
            #     cv2.waitKey(0)

            for j in range(self.num_classes):
                inds = np.where(labels == j)[0]
                if len(inds) == 0:
                    self.all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                    continue
                c_bboxes = bboxes[inds]
                c_scores = scores[inds]
                c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
                self.all_boxes[j][i] = c_dets
            
            print('Inference: {} / {}'.format(i+1, self.num_images), end='\r')
        
    def load_gt(self, classname):
        npos = 0
        gts = {}

        self.imgsetpath = os.path.join(self.data_dir, self.image_sets+'.txt')
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        
        for imagename in imagenames:
            annopath = self.parse_rec(os.path.join(self.data_dir, 'Annotations', '%s.txt')%(imagename))
            bboxes = [ins for ins in annopath if ins['name'] == classname]

            bbox = np.array([x['bbox'] for x in bboxes])
            difficult = np.array([x['difficult'] for x in bboxes]).astype(bool)
            det = [False] * len(bboxes)

            npos = npos + sum(~difficult)
            
            gts[imagename] = {'bbox': bbox,
                              'difficult': difficult,
                              'det': det}
        
        return gts, npos

    def load_dets(self, classname):
        image_ids = []
        confidence = []
        bboxes = []

        class_index = self.class_names.index(classname)
        for im_ind, dets in enumerate(self.all_boxes[class_index]):
            image_id = self.dataset.ids[im_ind][1]
            for k in range(dets.shape[0]):
                image_ids.append(image_id)
                confidence.append(dets[k, -1])
                bboxes.append([dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]])
       
        return {
            'image_ids': np.array(image_ids),
            'confidence': np.array(confidence),
            'bboxes': np.array(bboxes)
        }

    def eval(self, model):
        self.inference(model)
        print('\n~~~~~~~~')
        print('Results:')

        aps = []
        
        for cls_ind, cls_name in enumerate(self.class_names):
            dets = self.load_dets(cls_name)
            gts, npos = self.load_gt(cls_name)

            if len(dets['bboxes']):
                sorted_index = np.argsort(-dets['confidence'])
                sorted_image_ids = dets['image_ids'][sorted_index]
                sorted_confidence = dets['confidence'][sorted_index]
                sorted_bboxes = dets['bboxes'][sorted_index, :].astype(float)
                
                tp = np.zeros(len(dets['bboxes']))
                fp = np.zeros(len(dets['bboxes']))

                for index, box in enumerate(sorted_bboxes):
                    gt_dic = gts[sorted_image_ids[index]]
                    gt_boxes = gt_dic['bbox'].astype(float)
                    if gt_boxes.size > 0:
                        x_min = np.maximum(gt_boxes[:, 0], box[0])
                        y_min = np.maximum(gt_boxes[:, 1], box[1])
                        x_max = np.minimum(gt_boxes[:, 2], box[2])
                        y_max = np.minimum(gt_boxes[:, 3], box[3])

                        w_intersect = np.maximum(x_max - x_min, 0.)
                        h_intersect = np.maximum(y_max - y_min, 0.)

                        dt_area = (box[2] - box[0]) * (box[3] - box[1])
                        gt_area = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])

                        area_intersect = w_intersect * h_intersect
                        area_union = gt_area + dt_area - area_intersect
                        ious = area_intersect / np.maximum(area_union, 1e-10)

                        max_iou, max_index = np.max(ious), np.argmax(ious)

                        if max_iou > self.ovthresh and gt_dic['det'][max_index] != 1:
                            tp[index] = 1
                            gt_dic['det'][max_index] = 1
                        else:
                            fp[index] = 1 
                    else:
                        fp[index] = 1 

                # compute precision recall
                fp = np.cumsum(fp)
                tp = np.cumsum(tp)

                rec = tp / float(npos)
                # avoid divide by zero in case the first detection matches a difficult
                # ground truth
                tp = np.nan_to_num(tp, nan=0.0)
                fp = np.nan_to_num(fp, nan=0.0)
                prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

                ## 插值的P-R曲线
                rec_interp=np.linspace(0, 1, self.recall_thre) #101steps, from 0% to 100% 
                prec = np.interp(rec_interp, rec, prec, right=0)

                ap = self.voc_ap(rec_interp, prec)
            else:
                rec = 0.
                prec = 0.
                ap = 0.

            aps += [ap]
            
            print('{:<12} :     {:.3f}'.format(cls_name, ap))
            # break
        self.map = np.mean(aps)
        print('')
        print('~~~~~~~~')
        print('Mean AP = {:.4f}%'.format(np.mean(aps)*100))
        print('~~~~~~~~')
        print('')

        return self.map

if __name__ == "__main__":
    args = parse_args()

    if args.cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print('use cuda')
    else:
        device = torch.device('cpu')

    val_trans = Augmentation(args.image_size, args.data_augment, is_train=False)
    val_dataset = FACEDataset(data_dir     = args.data_root,
                             image_sets   = args.datasets_val,
                             transform    = val_trans,
                             is_train     = False,
                             classnames = args.class_names
                             )

    model = YOLOv3(device  =device,
                   backbone = args.backbone,
                   image_size   =args.image_size,
                   nms_thresh   =args.threshold_nms,
                   anchor_size = args.anchor_size,
                   num_classes  =args.classes_number,
                   conf_thresh  =args.threshold_conf,
                   boxes_per_cell=args.boxes_per_cell
                   ).to(device)
    

    ckpt_path = os.path.join(os.getcwd(), 'log', args.weight)
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=False)["model"]
    model.load_state_dict(state_dict)
    model.trainable = False
    model.eval()
    
    evaluator = Evaluator(
        device   =device,
        data_dir = args.data_root,
        dataset  = val_dataset,
        image_sets = args.datasets_val,
        ovthresh    = args.threshold_nms,                        
        class_names = args.class_names,
        recall_thre = args.threshold_recall,
        )

   # VOC evaluation
    map = evaluator.eval(model)