import os
import cv2
import torch
import pickle
import argparse
import numpy as np
from typing import List
from config import parse_args
from model.yolov1 import YOLOv1
import xml.etree.ElementTree as ET
from dataset.voc import VOCDataset
from dataset.augment import Augmentation

def rescale_bboxes(bboxes, origin_size, ratio):
    # rescale bboxes
    if isinstance(ratio, float):
        bboxes /= ratio
    elif isinstance(ratio, List):
        bboxes[..., [0, 2]] /= ratio[0]
        bboxes[..., [1, 3]] /= ratio[1]
    else:
        raise NotImplementedError("ratio should be a int or List[int, int] type.")

    # clip bboxes
    bboxes[..., [0, 2]] = np.clip(bboxes[..., [0, 2]], a_min=0., a_max=origin_size[0])
    bboxes[..., [1, 3]] = np.clip(bboxes[..., [1, 3]], a_min=0., a_max=origin_size[1])

    return bboxes

class VOCEvaluator():
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
        self.image_sets = image_sets[0],
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
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text),
                                int(bbox.find('ymin').text),
                                int(bbox.find('xmax').text),
                                int(bbox.find('ymax').text)]
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
  
    def inference(self, model, result_path):
        for i in range(self.num_images):
            img, target, deltas = self.dataset.__getitem__(i)
            orig_h, orig_w = img.shape[1:]

            # preprocess
            img = img.unsqueeze(0).to(self.device)

            # forward
            outputs = model(img)
            scores = outputs['scores']
            labels = outputs['labels']
            bboxes = outputs['bboxes']

            # rescale bboxes
            bboxes = rescale_bboxes(bboxes, [orig_w, orig_h], deltas)

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

        self.imgsetpath = os.path.join(self.data_dir, 'VOC'+self.image_sets[0][0], 'ImageSets', 'Main', self.image_sets[0][1] + '.txt')
        with open(self.imgsetpath, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        
        for imagename in imagenames:
            annopath = self.parse_rec(os.path.join(self.data_dir, 'VOC'+self.image_sets[0][0], 'Annotations', '%s.xml')%(imagename))
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

    def evaluate(self, model, result_path):
        self.inference(model, result_path)
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
    else:
        device = torch.device('cpu')

    val_transformer = Augmentation(args.img_size, args.data_augmentation, is_train=False)


    dataset = VOCDataset(
                         data_dir     = os.path.join(args.root, args.data),
                         image_sets   = args.val_sets,
                         transform    = val_transformer,
                         is_train     = False,
                         )

    model = YOLOv1(args = args, 
                   device = device,
                   trainable = False,
                   nms_thresh = args.nms_thresh,
                   conf_thresh = args.conf_thresh)
    weight_path = os.path.join(args.root, args.project, 'results', args.weight)
    checkpoint = torch.load(weight_path, map_location='cpu')
    checkpoint_state_dict = checkpoint["model"]
    print(checkpoint["mAP"])
    import time
    time.sleep(10)
    
    model.load_state_dict(checkpoint_state_dict)
    model.to(device).eval()
    

    evaluator = VOCEvaluator(
        device=device,
        data_dir = os.path.join(args.root, args.data),
        dataset = dataset,
        image_sets = args.val_sets,
        ovthresh = args.threshold,                        
        class_names = args.class_names,
        recall_thre = args.recall_thr,
        )

   # VOC evaluation
    map = evaluator.evaluate(model, result_path = weight_path.replace('.pth', ''))