import os
import sys
cur = os.path.dirname(os.path.abspath(__file__))
pro_path = os.path.abspath(os.path.join(cur, '..'))
sys.path.append(pro_path)

import onnx
import torch
from model.yolo import YOLO
from config import parse_args

def export(input, model, weight):
    model.deploy = True
    model.trainable = False
    pt_onnx = weight.replace('.pth', '.onnx')

    state_dict = torch.load(os.path.join('log', weight), 
                            map_location = 'cpu', 
                            weights_only = False)
    model.load_state_dict(state_dict.get("model", state_dict))
    
    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            pt_onnx,
            opset_version=11,
            input_names=['input'],
            output_names=['output'])

    # 添加中间层特征尺寸
    onnx_model = onnx.load(pt_onnx) 
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), pt_onnx)

    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect")
    else: 
        print("Model correct")

if __name__ == "__main__":
    parser, args = parse_args()

    x = torch.randn(1, 3, 608, 608)

    model = YOLO(device = torch.device('cpu'),
                 trainable =  False,
                 backbone = args.backbone,
                 anchor_size = args.anchor_size,
                 num_classes = args.num_classes,
                 nms_threshold = args.nms_threshold,
                 boxes_per_cell = args.boxes_per_cell,
                 confidence_threshold = args.confidence_threshold
                 ).eval()
    
    
    export(x, model, args.model_weight_path)