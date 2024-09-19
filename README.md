# Yolo_V5 (You only look once)

- Yolo_v5 pytorch源码，含训练、推理、评测、部署脚本
- 细节请参考Yolo_v4论文描述：
https://arxiv.org/abs/2004.10934 (v4)

## Train Tricks

|Epoch|BatchSize|Pretrain|WarmUp|Datasets|TrainSize|ValSize|Augment |
|:---:|:---:    |:---:   |:---: |:---:   |:---:    |:---:  |:---:   |
|300  |64       |CoCo    |3     |VOC20   |16553    |4952   |hsv+Ablu|

|NMS_TH.|EvalIou_TH.|Confidence_TH.|Optimizer|Grad_accu|Lr_sche|LearningRate|Model_Ema|
|:---:  |:---:      |:---:         |:---:    |:---:    |:---:  |:---:       |:---:    |
|0.7    |0.5        |0.001         |adamw    |1        |linear |0.001       |True     |

## Results:

|Backbone    |Size |Params|FLOPs |Model  |Params|FLOPs  |Mosaic|Mixup|mAP  |
|:---:       |:---:|:---: |:---: |:---:  |:---: |:---:  |:---: |:---:|:---:|
|cspdarknet_n|512  | 0.89M| 1.82G|Yolov5n| 2.26M|  4.48G|      |     |     |
|cspdarknet_t|512  | 2.00M| 3.88G|Yolov5t| 5.05M|  9.82G|      |     |     |
|cspdarknet_s|512  | 3.56M| 6.72G|Yolov5s| 8.96M| 17.22G|1.0   |     |     |
|cspdarknet_m|512  |10.88M|20.79G|Yolov5m|25.32M| 47.32G|      |     |     |
|cspdarknet_l|512  |24.45M|47.37G|Yolov5l|54.20M| 99.82G|      |     |     |
|cspdarknet_x|512  |46.20M|90.49G|Yolov5x|99.06M|180.73G|      |     |     |

<table>
<tr><th>Yolo_v4_Tiny</th> <th>Yolo_v4_CSPDarknet_Tiny</th></tr>
<tr>

<td>
    
|ClassNames |AP   |
|--         |--   |
|mAP        |0.672|

</td>
<td>
    
|ClassNames |AP   |
|--         |--   |
|mAP        |0.638|

</td>

</tr> 
</table>

<video src="" 
       controls 
       width="100%" 
       height="auto" 
       style="max-width: 720px; height: auto; display: block; object-fit: contain;">
</video>