import os
import sys
import onnx
import torch
sys.path.append('../')
from config import parse_args
from model.build import build_yolo

def export(input, model, weight_name):
    weight_path = os.path.join(os.getcwd().replace('deploy', 'log'), weight_name)
    pt_onnx = weight_path.replace('.pth', '.onnx')

    state_dict = torch.load(weight_path, 
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
    args = parse_args()
    args.resume_weight_path = 'None'

    x = torch.randn(1, 3, 512, 512)

    model = build_yolo(args, torch.device('cpu'), False)
    model = model.eval()
    model.deploy = True
   
    export(x, model, args.model_weight_path)