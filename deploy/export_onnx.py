import os
import sys
import onnx
import torch
import argparse
sys.path.append('../')
from model.yolov3 import Yolo_V2

def export(input, model, pt_path, onnx_version):
    model.eval()
    model.deploy = True
    model.trainable = False

    ckpt = os.getcwd().replace('deploy', pt_path)
    state_dict = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state_dict["model"])

    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            onnx_version+".onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'])

    # 添加中间层特征尺寸
    onnx_model = onnx.load(onnx_version+".onnx") 
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), onnx_version+".onnx")

    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect")
    else: 
        print("Model correct")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--cuda',           default=False,  help='Weather use cuda.')
    parser.add_argument('--batch_size',     default=1,      help='The batch size used by a single GPU during training')
    parser.add_argument('--image_size',     default=416,    help='input image size')
    parser.add_argument('--num_classes',    default=20,     help='The number of the classes')
    parser.add_argument('--boxes_per_cell', default=5,      help='The number of the boxes in one cell')
    parser.add_argument('--threshold_conf', default=0.3,    help='confidence threshold')
    parser.add_argument('--threshold_nms',  default=0.5,    help='NMS threshold')
    parser.add_argument('--classes_number', default=20,     help='The number of the classes')

    args = parser.parse_args()

    version = "yolo_v2"
    pt_path = "log/0.pth"
    device = torch.device('cpu')

    x = torch.randn(1, 3, 416, 416)

    model = Yolo_V2(device = device,
            image_size=args.image_size,
            nms_thresh=args.threshold_nms,
            num_classes=args.classes_number,
            conf_thresh = args.threshold_conf,
            boxes_per_cell=args.boxes_per_cell
            ).to(device)
    
    export(x, model, pt_path, version)