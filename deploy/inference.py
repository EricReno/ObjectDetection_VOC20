import os
import cv2
import time
import numpy
import argparse
import onnxruntime

def parse_args():
    parser = argparse.ArgumentParser(description='Infer')
    parser.add_argument('--cuda', default=False, help='Weather use cuda.')
    
    parser.add_argument('--onnx', default='yolo_v1.onnx', help='The onnx file which will be used.')
    parser.add_argument('--image_path', default='images', help='The root directory where data are stored')
    parser.add_argument('--image_size', default = 640,    help='input image size')
    parser.add_argument('--confidece', default = 0.3,     help='The confidence threshold of predicted objects')
    parser.add_argument('--nms_thresh', default = 0.5,    help='NMS threshold')
    
    parser.add_argument('--class_names', default= ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 
                           'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 
                           'sofa', 'train', 'tvmonitor'], help= 'The category of predictions that the model can cover')
    parser.add_argument('--class_colors', default= {
                                                    'aeroplane': (255, 0, 0),       # 蓝色
                                                    'bicycle': (0, 255, 0),         # 绿色
                                                    'bird': (0, 0, 255),            # 红色
                                                    'boat': (255, 255, 0),          # 青色
                                                    'bottle': (255, 0, 255),        # 洋红
                                                    'bus': (0, 255, 255),           # 黄色
                                                    'car': (128, 0, 128),           # 紫色
                                                    'cat': (0, 128, 128),           # 深绿色
                                                    'chair': (128, 128, 0),         # 橄榄绿
                                                    'cow': (64, 64, 64),            # 灰色
                                                    'diningtable': (0, 64, 128),    # 深蓝色
                                                    'dog': (128, 64, 0),            # 棕色
                                                    'horse': (0, 128, 64),          # 暗绿色
                                                    'motorbike': (128, 128, 255),   # 浅蓝色
                                                    'person': (64, 0, 64),          # 紫红色
                                                    'pottedplant': (128, 0, 0),     # 深红色
                                                    'sheep': (0, 128, 255),         # 天蓝色
                                                    'sofa': (128, 255, 128),        # 浅绿色
                                                    'train': (255, 128, 0),         # 橙色
                                                    'tvmonitor': (64, 255, 64)      # 浅绿色
                                                }, help= 'The category of predictions that the model can cover')
    return parser.parse_args()

## basic NMS
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
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
        xx1 = numpy.maximum(x1[i], x1[order[1:]])
        yy1 = numpy.maximum(y1[i], y1[order[1:]])
        xx2 = numpy.minimum(x2[i], x2[order[1:]])
        yy2 = numpy.minimum(y2[i], y2[order[1:]])

        w = numpy.maximum(1e-10, xx2 - xx1)
        h = numpy.maximum(1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = numpy.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

def preinfer(image_path, image_size):
    image = cv2.imread(image_path)

    ratio = [image_size/image.shape[1], image_size/image.shape[0]]

    output = cv2.resize(image, (image_size, image_size))
    output = output.astype(numpy.float32)
    output = output.transpose([2, 0, 1])
    output /= 255.
    output = numpy.expand_dims(output, 0)

    return  image, output, ratio

def infer(input, onnx, cuda):
    start = time.time()

    if cuda:
        providers = [('CUDAExecutionProvider', {
            'device_id': 0
        })]
    else:
        providers = [('CPUExecutionProvider', {})]
    
    session = onnxruntime.InferenceSession(onnx, providers=providers)
    
    output = session.run(['output'], {'input': input})

    end = time.time() - start
    print("Inference time (Hz):", 1 / end)

    return output

def postinfer(input, ratio, image_size, class_names, conf_thresh, nms_thresh):
    bboxes = input[0][:, :4]
    scores = input[0][:, 4:]

    labels = numpy.argmax(scores, axis=1)
    scores = scores[(numpy.arange(scores.shape[0]), labels)]
        
    # 初筛: confidecn threshold 
    keep = numpy.where(scores >= conf_thresh)
    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    # 放缩：放缩到原图&边界值处理
    bboxes[..., [0, 2]] /= ratio[0]
    bboxes[..., [1, 3]] /= ratio[1]
    bboxes[..., [0, 2]] = numpy.clip(bboxes[..., [0, 2]], a_min=0., a_max=(image_size/ratio[0]))
    bboxes[..., [1, 3]] = numpy.clip(bboxes[..., [1, 3]], a_min=0., a_max=(image_size/ratio[1]))

    # NMS: Non-Maximum Suppression
    keep = numpy.zeros(len(bboxes), dtype=numpy.int32)
    for i in range(len(class_names)):
        inds = numpy.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep[inds[c_keep]] = 1
    keep = numpy.where(keep > 0)
    scores = scores[keep]
    labels = labels[keep]
    bboxes = bboxes[keep]

    return labels, scores, bboxes

if __name__ == "__main__":
    args = parse_args()

    images_list = [os.path.join(os.path.abspath(args.image_path), _) for _ in os.listdir(args.image_path)]
    for image in images_list:

        start_time = time.time()
        ## TODO ONE
        image, infer_input, ratio = preinfer(image, args.image_size)

        ## TODO TWO
        postinfer_input = infer(infer_input, args.onnx, args.cuda) # 400*(4+20)

        ## TODO THREE
        labels, scores, bboxes = postinfer(postinfer_input, ratio, args.image_size, 
                                           args.class_names, args.confidece, args.nms_thresh)

        end_time = time.time()
        ## TODO FOUR
        for i, bbox in enumerate(bboxes):
            score = scores[i]
            label = labels[i]
            
            label_name = args.class_names[label]
            bbox = [int(point) for point in bbox]

            cv2.rectangle(image, (bbox[0], bbox[1]),  (bbox[2], bbox[3]), args.class_colors[label_name], 1)
            
            text = "%s:%s"%(label_name, str(round(float(score), 2)))
            (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            cv2.rectangle(image, (bbox[0], bbox[1]),  (bbox[0] + w, bbox[1] + h), args.class_colors[label_name], -1) 
            cv2.putText(image, text, (bbox[0], bbox[1]+h), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            
            text = "fps:%s"%(str(round(1 / (end_time - start_time), 2)))
            (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 1)
            cv2.rectangle(image, (0, 0), (w, h), (255, 255, 255), -1) 
            cv2.putText(image, text, (0, h), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)

        cv2.imshow('image', image)

        # 退出循环的按键（通常是'q'键）  
        if cv2.waitKey(2) == ord('q'):  
            break