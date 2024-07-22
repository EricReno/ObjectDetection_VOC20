import os
import sys
import onnx
import torch
sys.path.append('../')
from config import parse_args
from model.yolov1 import YOLOv1

def export(input, model, pt_path, onnx_name):
    model.eval()
    model.deploy = True

    ckpt = os.getcwd().replace('deploy', pt_path)
    state_dict = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(state_dict["model"])

    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            "yolo_v1.onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'])

    # 添加中间层特征尺寸
    onnx_model = onnx.load(onnx_name) 
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), onnx_name)

    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect")
    else: 
        print("Model correct")

if __name__ == "__main__":
    version = "yolo_v1"
    pt_path = "results/84.pth"

    args = parse_args()

    x = torch.randn(1, 3, 640, 640)

    if version == "yolo_v1":
        model = YOLOv1(args = args, 
               device = torch.device('cpu'),
               trainable = False,
               nms_thresh = args.nms_thresh,
               conf_thresh = args.conf_thresh)

    export(x, model, pt_path, version+".onnx")