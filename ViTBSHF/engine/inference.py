# ==============================================================================
# HBF-ViT 行人重识别系统 - 独立推理与学术评估脚本 (Inference & Evaluation)
# ==============================================================================

import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# 引入核心组件
from utils.logger import setup_logger
from utils.iotools import load_config
from utils.checkpoint import load_checkpoint
from datasets.loader import make_dataloader
from modeling.modeling_builder import make_model
from utils.metrics import R1_mAP_eval

def plot_cmc_curve(cmc, output_dir, max_rank=20):
    """
    绘制并保存 CMC (Cumulative Matching Characteristics) 曲线
    
    学术支撑: CMC 曲线是衡量 ReID 系统检索精度的直观标准。
    横坐标为 Rank-k，纵坐标为在 Top-k 结果中找到至少一个正确匹配的概率。
    """
    ranks = np.arange(1, max_rank + 1)
    cmc_to_plot = cmc[:max_rank]

    plt.figure(figsize=(10, 6))
    plt.plot(ranks, cmc_to_plot, marker='o', linestyle='-', color='b', label='ViT-HBF')
    
    # 图表格式化，达到论文发表标准
    plt.title('CMC Curve', fontsize=16, fontweight='bold')
    plt.xlabel('Rank', fontsize=14)
    plt.ylabel('Matching Rate (%)', fontsize=14)
    plt.xticks(np.arange(1, max_rank + 1, step=max(1, max_rank//10)))
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=12)
    plt.tight_layout()

    # 保存高斯清晰度的图表，适用于 CVPR/ICCV 等顶级会议论文插入
    save_path = os.path.join(output_dir, 'cmc_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    return save_path

def main():
    parser = argparse.ArgumentParser(description="HBF-ViT ReID Inference")
    parser.add_argument("--config_file", default="configs/msmt_hbf_vit.yml", help="配置文件路径", type=str)
    parser.add_argument("--weights", required=True, help="待测试的模型权重路径 (.pth)", type=str)
    args = parser.parse_args()

    # 1. 加载配置与初始化日志
    cfg = load_config(args.config_file)
    # 推理日志可以直接输出到终端，同时保存在 OUTPUT_DIR
    logger = setup_logger("reid.inference", cfg.OUTPUT_DIR, distributed_rank=0)
    logger.info(f"==> 启动推理评估引擎，加载配置: {args.config_file}")

    # 2. 硬件设备设定
    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")
    logger.info(f"==> 评估设备: {device}")

    # 3. 准备数据加载器 (重点关注 val_loader 和 num_query)
    # 测试期关闭 shuffle，保持 Query 和 Gallery 严格的拼接顺序
    _, val_loader, num_query, num_classes, num_cameras = make_dataloader(cfg)
    logger.info(f"==> 测试集加载完成: Query 数量={num_query}, 总类别数={num_classes}")

    # 4. 实例化模型
    # 注意: 推理时不需要传递 view_num，除非模型需要该先验信息
    model = make_model(cfg, num_classes=num_classes, camera_num=num_cameras, view_num=0)
    model.to(device)

    # 5. 严格加载模型权重
    logger.info(f"==> 正在加载模型权重: {args.weights}")
    load_checkpoint(args.weights, model)
    
    # 切换至评估模式，冻结 BatchNorm 等动态统计层，防止统计量偏移 (Covariate Shift)
    model.eval()

    # 6. 实例化特征评估器
    # 引用自 utils.metrics.R1_mAP_eval
    # 自动处理了 "剔除同一摄像头下相同身份图像" 这一严苛的学术标准
    evaluator = R1_mAP_eval(
        num_query=num_query, 
        max_rank=50, 
        feat_norm=cfg.TEST.FEAT_NORM, 
        reranking=cfg.TEST.RE_RANKING
    )
    evaluator.reset()

    # 7. 启动特征提取循环
    logger.info("==> 开始全量提取测试集特征 (Extracting Features)...")
    
    # torch.no_grad() 是关键：切断计算图，大幅降低推理显存占用并提升速度
    with torch.no_grad():
        # 使用 tqdm 包装 val_loader 以显示美观的进度条
        for batch_idx, batch_data in enumerate(tqdm(val_loader, desc="Feature Extraction")):
            # 兼容不同 collate_fn 的返回长度
            if len(batch_data) == 4:
                imgs, pids, camids, img_paths = batch_data
            else:
                imgs, pids, camids = batch_data[:3]
                
            imgs = imgs.to(device)
            
            # 提取表观特征向量 (经过 HBF 模块净化和 BNNeck 投影后的特征)
            features = model(imgs)
            
            # 将当前 Batch 的结果推入评估器
            evaluator.update((features, pids, camids))

    # 8. 核心距离矩阵计算与指标输出
    logger.info("==> 特征提取完毕，正在计算距离矩阵与 CMC/mAP 指标...")
    cmc, mAP, mINP = evaluator.compute()

    logger.info(f"================== 最终学术评估结果 ==================")
    logger.info(f"mAP      : {mAP:.2%}")
    logger.info(f"mINP     : {mINP:.2%} (衡量最难正样本的逆惩罚指标)")
    logger.info(f"Rank-1   : {cmc[0]:.2%}")
    logger.info(f"Rank-5   : {cmc[4]:.2%}")
    logger.info(f"Rank-10  : {cmc[9]:.2%}")
    logger.info(f"Rank-20  : {cmc[19]:.2%}")
    logger.info(f"=====================================================")

    # 9. 绘制并保存 CMC 曲线
    logger.info("==> 正在生成 CMC 曲线图表...")
    curve_path = plot_cmc_curve(cmc, cfg.OUTPUT_DIR, max_rank=20)
    logger.info(f"==> CMC 曲线已成功保存至: {curve_path}")


if __name__ == "__main__":
    main()