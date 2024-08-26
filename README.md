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
| Size  |  BS | Pretrain| Epoch| Obj | Cls |  Box | NMS | Confidence| APT |
| :---: |:---:|  :---:  | :---:|:---:|:---:| :---:|:---:| :---:    | :---:|
|608x608|  24 |   CoCo  |  160 | 1.0 | 1.0 | 5.0  | 0.5 |  0.3     | SGD  |

|DataAugmentation    |
|        :---:       |
|RandomSaturationHue |
|RandomContrast      |
|RandomBrightness    |
|RandomSampleCrop    |
|RandomExpand        |
|RandomHorizontalFlip|


## VOC20_val:
| TAG  |  Size|    mAP    |    GFLOPs     |Params |Pt_Size| FPS |
| :---: |   :---:   | :---:   |  :---:  |:---:  |:---:  |:---:  |
|yolo v1|   608   |67.47%  |         | |||
|yolo_v3_tiny|   608   |56.17%  |   5.18      | 2.43| 19M|75.44(1050Ti)|
|yolo_v3_Darknet53|   608   |75.71%  |  133.40      | 57.43| 442M|10.32(1050Ti)|

## Demo
https://github.com/user-attachments/assets/4996663b-9bbd-4c83-a535-812f4e401b5d

