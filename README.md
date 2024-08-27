# Hello, You Only Look Once -- VOC20

- 传送门：
https://arxiv.org/abs/1506.02640 (Yolo v1)
https://arxiv.org/abs/1612.08242 (Yolo v2)
https://arxiv.org/abs/1804.02767 (Yolo v3)
https://arxiv.org/abs/2004.10934 (Yolo v4)

## 数据集: Fire_Smoke
- **test**: 300
- **train**: 3000
- **CLASSES_NAMES**: 

|          |          | 
| :------: | :------: |
|   fire   |   smoke  | 


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


## Test:
| TAG  |  Size|    mAP    |    GFLOPs     |Params |Pt_Size| FPS |
| :---: |   :---:   | :---:   |  :---:  |:---:  |:---:  |:---:  |
|yolo_tiny|   608   |%  |         | | |(1050Ti)|
|yolo_Darknet53|   608   |%  |        || |(1050Ti)|

## Demo
<video src="" 
       controls 
       width="100%" 
       height="auto" 
       style="max-width: 720px; height: auto; display: block; object-fit: contain;">
</video>

