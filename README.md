# Yolo_V3 (You only look once)

- yolo系列的pytorch源码学习，已更新至yolov4，含训练、推理、评测、部署脚本
- 论文传送门：

https://arxiv.org/abs/2004.10934 (v4)

## 数据集: VOC 
- **test**: (VOC2007, test) : 4952
- **train**: (VOC2007, trainval), (VOC2012, trainval) : 16553
- **CLASSES_NAMES**:

|             |          |         |           |           |
| :---------: | :------: | :-----: | :-------: | :-------: |
|  aeroplane  | bicycle  |  bird   |   boat    | bottle    |
|     bus     |   car    |  cat    |  chair    | cow       |
| diningtable |   dog    | horse   | motorbike | person    |
| pottedplant |  sheep   |  sofa   |  train    | tvmonitor |

- **官方网址** 

    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html
    
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html


## 通用设置

|:---:               |
|RandomSaturationHue |
|RandomContrast      |
|RandomBrightness    |
|RandomSampleCrop    |
|RandomExpand        |
|RandomHorizontalFlip|

|BS   |Pretrained|Epoch|Obj_loss|Cls_loss|Box_loss|NMS_th|Confidence|APT  |LearningRate|Lr_scheduler|DataAugmentation|
|:---:|:---:     |:---:|:---:   |:---:   |:---:   |:---: |:---:     |:---:|:---:       |:---:       |:---:       |
|  64 |CoCo      |160  |1.0     | 1.0    | 5.0    |0.5   |0.3       |SGD  |0.01        |linear      |SSD|

## Results:

|Backbone    |Size |Params|FLOPs |Speedcpu|Model  |Params|FLOPs |
|:---:       |:---:|:---: |:---: |:---:   |:---:  |:---: |:---: |
|cspdarknet_n|512  | 0.89M| 1.82G| 43.96ms|Yolov5n| 2.26M| 4.48G|
|cspdarknet_t|512  | 2.00M| 3.88G| 48.63ms|Yolov5t| 5.05M| 9.82G|
|cspdarknet_s|512  | 3.56M| 6.72G| 63.03ms|Yolov5s| 8.96M|17.22G|
|cspdarknet_m|512  |10.88M|20.79G|117.68ms|Yolov5m|25.32M|47.32G|
|cspdarknet_l|512  |24.45M|47.37G|214.03ms|Yolov5l|54.20M|99.82G|
|cspdarknet_x|512  |46.20M|90.49G|366.54ms|Yolov5x|99.06M|180.73G|

<table>
<tr><th>Yolo_v4_Tiny</th> <th>Yolo_v4_CSPDarknet_Tiny</th></tr>
<tr>
<td>
    
|ClassNames |AP   |
|--         |--   |
|aeroplane  |0.681|
|bicycle    |0.706|
|bird       |0.671|
|boat       |0.605|
|bottle     |0.370|
|bus        |0.755|
|car        |0.758|
|cat        |0.771|
|chair      |0.482|
|cow        |0.699|
|diningtable|0.637|
|dog        |0.729|
|horse      |0.766|
|motorbike  |0.740|
|person     |0.642|
|pottedplant|0.328|
|sheep      |0.602|
|sofa       |0.771|
|train      |0.753|
|tvmonitor  |0.677|
|mAP        |0.672|

</td>
<td>
    
|ClassNames |AP   |
|--         |--   |
|           |     |
|aeroplane  |0.745|
|bicycle    |0.795|
|bird       |0.780|
|boat       |0.683|
|bottle     |0.473|
|bus        |0.872|
|car        |0.797|
|cat        |0.881|
|chair      |0.609|
|cow        |0.824|
|diningtable|0.749|
|dog        |0.874|
|horse      |0.855|
|motorbike  |0.790|
|person     |0.688|
|pottedplant|0.433|
|sheep      |0.676|
|sofa       |0.862|
|train      |0.832|
|tvmonitor  |0.742|

|mAP        |0.638|

</td>
</tr> 
</table>

<video src="https://github.com/user-attachments/assets/d5811825-8c58-4f0f-9067-a79d0c9966dc" 
       controls 
       width="100%" 
       height="auto" 
       style="max-width: 720px; height: auto; display: block; object-fit: contain;">
</video>