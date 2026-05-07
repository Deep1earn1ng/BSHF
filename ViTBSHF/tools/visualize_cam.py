# tools/visualize_cam.py
# ==============================================================================
# HBF-ViT 学术分析工具 - 独立 Grad-CAM 可视化脚本
# 目的: 生成论文所需的注意力热力图对比 (Ablation Visualization)
# ==============================================================================

import argparse
import os
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 导入内部组件
from utils.iotools import load_config
from modeling.modeling_builder import make_model
from utils.checkpoint import load_checkpoint
from datasets.make_dataloader import make_dataloader


class ReIDFeatureTarget:
    """ 学术级自定义激活目标：以模型输出的特征向量自身作为反向传播的 Target """
    def __init__(self, feature_vector):
        self.feature_vector = feature_vector

    def __call__(self, model_output):
        if isinstance(model_output, (list, tuple)):
            model_output = model_output[0]
        feat = self.feature_vector
        if isinstance(feat, (list, tuple)):
            feat = feat[0]
        return (model_output * feat).sum()

def reshape_transform_vit(tensor, height=24, width=8):
    """ 
    ViT 专用 Hook 转换器
    作用: 剔除 [CLS] Token，将 192 个 Patch Tokens 还原为 2D 空间特征图
    """
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def main():
    parser = argparse.ArgumentParser(description="HBF-ViT Grad-CAM Visualization")
    parser.add_argument("--config_file", default="configs/market_hbf_vit.yml", help="配置文件路径")
    parser.add_argument("--weights", required=True, help="模型权重路径 (.pth)")
    parser.add_argument("--img_path", required=True, help="待分析的原始图像路径")
    parser.add_argument("--save_path", default="cam_output.jpg", help="热力图保存路径")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config_file)

    # 1. 动态读取当前数据集的维度信息
    print("📦 [分析] 正在解析数据集维度先验...")
    _, _, _, num_classes, num_cameras = make_dataloader(cfg)

    # 2. 准备模型 (动态传入真实类别和相机数)
    print(f"🧠 [分析] 加载网络架构与权重: {args.weights}")
    model = make_model(cfg, num_classes=num_classes, camera_num=num_cameras, view_num=0)
    model.to(device)
    load_checkpoint(args.weights, model)
    model.eval()

    # 2. 图像预处理 (严格对齐训练时的 Normalization)
    transform = transforms.Compose([
        transforms.Resize((384, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    raw_img = Image.open(args.img_path).convert('RGB')
    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    cv_img = cv2.imread(args.img_path)
    cv_img = cv2.resize(cv_img, (128, 384))
    rgb_img = np.float32(cv_img) / 255.0

    # 3. 指定目标层 (ViT-Base 的最后一个 Block)
    target_layers = [model.base.blocks[-1].norm1]

    # 4. 提取目标特征
    with torch.no_grad():
        target_feat = model(input_tensor)
        if isinstance(target_feat, (list, tuple)):
            target_feat = target_feat[0]

    # 5. 实例化 CAM
    cam = GradCAM(
        model=model,
        target_layers=target_layers,
        use_cuda=torch.cuda.is_available(),
        reshape_transform=reshape_transform_vit
    )

    # 6. 生成并融合热力图
    targets = [ReIDFeatureTarget(target_feat)]
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=False)

    cv2.imwrite(args.save_path, visualization)
    print(f"✅ [分析] 热力图已成功保存至: {args.save_path}")

if __name__ == "__main__":
    main()