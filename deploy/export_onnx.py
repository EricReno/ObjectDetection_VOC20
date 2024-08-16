import os
import sys
import onnx
import torch
sys.path.append('../')
from config import parse_args
from model.yolov3 import YOLOv3

def export(input, model, pt_path, onnx_version):
    model.eval()
    model.deploy = True
    model.trainable = False

    ckpt = os.getcwd().replace('deploy', pt_path)
    state_dict = torch.load(ckpt, map_location='cuda', weights_only=False)
    model.load_state_dict(state_dict["model"])

    with torch.no_grad():
        torch.onnx.export(
            model,
            input,
            onnx_version+".onnx",
            opset_version=11,
            input_names=['input'],
            output_names=['output'],
            verbose=False)

    # 添加中间层特征尺寸
    onnx_model = onnx.load(onnx_version+".onnx") 
    
    # onnx.save(onnx.shape_inference.infer_shapes(onnx_model), onnx_version+".onnx")

    try: 
        onnx.checker.check_model(onnx_model) 
    except Exception: 
        print("Model incorrect")
    else: 
        print("Model correct")

if __name__ == "__main__":
    args = parse_args()

    version = "yolo_v3"
    pt_path = "log/151.pth"
    device = torch.device('cuda')

    x = torch.randn(1, 3, 608, 608).to(device)

    model = YOLOv3(device = device,
                   backbone = args.backbone,
                   image_size=args.image_size,
                   nms_thresh=args.threshold_nms,
                   anchor_size = args.anchor_size,
                   num_classes=args.classes_number,
                   conf_thresh = args.threshold_conf,
                   boxes_per_cell=args.boxes_per_cell
                   ).to(device)
    
    export(x, model, pt_path, version)