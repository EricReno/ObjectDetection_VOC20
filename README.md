# Hello, You Only Look Once -- VOC20
(You only live once)

- 传送门：https://arxiv.org/abs/1506.02640 (Yolo v1)

## 数据集: VOC 
- **test**: (VOC2007, test) : 4952
- **train**: (VOC2007, trainval), (VOC2012, trainval) : 16553
- **CLASSES_NAMES**: 

|             |          |         |           |           |
| :---------: | :------: | :-----: | :-------: | :-------: |
|  aeroplane  | bicycle  |  bird   |   boat    | bottle    |
|     bus     |   car    |  cat    |  chair    | cow       |
| diningtable |   dog    | horse   | motorbike | person    |
| pottedplant |  sheep   |  sofa   |  train    | tvmonitor |

- **官方网址** 
    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html


## 通用设置
| Size  |  BS | Pretrain| Epoch| Obj | Cls | Box | NMS_Thre | Conf_Thre| APT
| :---: |:---:|  :---:  | :---:|   :---: |:---:  | :---:      | :---:    | :---:    | :---:    |
|608x608|  24 |   CoCo  |  160 |   1.0 | 1.0  | 5.0        | 0.5      |  0.3     | SGD|

| Augmentation|
|   :---:     |
|RandomSaturationHue|
|RandomContrast|
|RandomBrightness|
|RandomSampleCrop|
|RandomExpand|
|RandomHorizontalFlip|


## DataAugment:
| TAG  |  Size|    mAP    |    GFLOPs     |Params |Pt_Size| FPS |
| :---: |   :---:   | :---:   |  :---:  |:---:  |:---:  |:---:  |
|yolo v1|   608   |67.47%  |         | |||
|yolo_v3_tiny|   608   |56.17%  |   5.18      | 2.43| 19M|75.44(1050Ti)|
|yolo_v3_Darknet53|   608   |....%  |  133.40      | 57.43| 442M|10.26(1050Ti)|

<img src="1.jpg">
