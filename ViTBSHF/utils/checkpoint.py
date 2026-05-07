import os
import math
import torch
import logging
import torch.nn.functional as F

def save_checkpoint(cfg, epoch, model, optimizer, scheduler, is_best_map=False, is_best_rank1=False):
    """保存模型与训练状态字典"""
    logger = logging.getLogger("reid.checkpoint")
    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'cfg': cfg,
        'seed': getattr(cfg.SOLVER, 'SEED', 1234)
    }

    latest_path = os.path.join(output_dir, f"{cfg.MODEL.NAME}_epoch_{epoch}.pth")
    torch.save(state, latest_path)

    if is_best_map:
        best_map_path = os.path.join(output_dir, f"TOP_SCORE_{cfg.MODEL.NAME}.pth")
        torch.save(state, best_map_path)
        logger.info(f"🏆 发现更高 mAP！已覆盖保存至: {best_map_path}")

    if is_best_rank1:
        best_r1_path = os.path.join(output_dir, f"TOP_RANK1_{cfg.MODEL.NAME}.pth")
        torch.save(state, best_r1_path)
        logger.info(f"🚀 发现更高 Rank-1！已覆盖保存至: {best_r1_path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """加载模型权重（支持断点续训）"""
    logger = logging.getLogger("reid.checkpoint")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ 未找到权重文件: {path}")

    logger.info(f"=> 正在加载权重: {path}")
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    model_state = checkpoint.get('state_dict', checkpoint)
    msg = model.load_state_dict(model_state, strict=False)
    logger.info(f"模型加载结果: {msg}")

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> 成功恢复优化器状态")

    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        logger.info("=> 成功恢复调度器状态")

    return checkpoint.get('epoch', 0)


def load_pretrain_vit(model, pretrain_path):
    """
    加载 ViT 预训练权重 (学术级增强版)
    具备：1. 前缀自适应映射 2. Positional Embedding 动态长宽比双三次插值能力
    """
    logger = logging.getLogger("reid.checkpoint")
    if not os.path.exists(pretrain_path):
        logger.warning(f"⚠️ 未找到预训练权重: {pretrain_path}，模型将从完全随机初始化开始！")
        return

    checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=True)
    if 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    elif 'model' in checkpoint:
        checkpoint = checkpoint['model']

    model_dict = model.state_dict()
    new_pretrained_dict = {}
    match_count = 0
    
    logger.info(f"=> 开始进行键值对齐与预训练加载: {pretrain_path}")

    for k, v in checkpoint.items():
        # [学术修复 1] 自动桥接前缀，解决嵌套模型导致键名匹配失败的问题
        new_k = 'base.' + k if not k.startswith('base.') else k
        
        # [学术修复 2] 动态拦截并重构位置编码 (Positional Embedding)
        if new_k == 'base.pos_embed' and new_k in model_dict:
            if v.shape != model_dict[new_k].shape:
                logger.info(f"🔄 检测到位置编码维度不一致，启动双三次插值 (Bicubic Interpolation): {v.shape} -> {model_dict[new_k].shape}")
                
                # 剥离 CLS Token
                cls_pos = v[:, 0:1, :]
                patch_pos = v[:, 1:, :]
                
                # 1. 动态推断【预训练模型】的网格形状 (h_pre, w_pre)
                num_patches_pretrained = patch_pos.shape[1]
                if int(math.sqrt(num_patches_pretrained)) ** 2 == num_patches_pretrained:
                    # 情况A: ImageNet 预训练 (正方形, 如 224x224 -> 14x14)
                    h_pre = w_pre = int(math.sqrt(num_patches_pretrained))
                elif num_patches_pretrained == 128:
                    # 情况B: ReID 专项预训练 (长方形，通常是 256x128 -> 16x8)
                    h_pre, w_pre = 16, 8
                else:
                    raise RuntimeError(f"❌ 未知的预训练Patch数: {num_patches_pretrained}，无法推导长宽比例。")

                # 重塑源空间
                patch_pos = patch_pos.reshape(1, h_pre, w_pre, -1).permute(0, 3, 1, 2)
                
                # 2. 推导当前 ReID 任务的【目标网格】
                target_num_patches = model_dict[new_k].shape[1] - 1
                if target_num_patches == 192:
                    # 标准 384x128 且步长 16x16 -> 24x8 的网格
                    target_h, target_w = 24, 8
                elif target_num_patches == 128:
                    # 标准 256x128 -> 16x8 的网格
                    target_h, target_w = 16, 8
                else:
                    # 通用推导逻辑：假设宽始终对应 128 分辨率的 8 个 Patch
                    target_w = 8
                    target_h = target_num_patches // 8
                
                # 3. 执行自适应长宽比的插值
                patch_pos = F.interpolate(
                    patch_pos, 
                    size=(target_h, target_w), 
                    mode='bicubic', 
                    align_corners=False
                )
                
                # 还原维度映射
                patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, target_num_patches, -1)
                
                # 缝合 CLS Token
                v = torch.cat((cls_pos, patch_pos), dim=1)

        # 严格过滤与合并
        if new_k in model_dict and v.shape == model_dict[new_k].shape:
            new_pretrained_dict[new_k] = v
            match_count += 1

    if match_count == 0:
        logger.error("❌ 严重警告：加载的预训练张量为 0！键名前缀不匹配，模型目前为盲人状态！")
    else:
        # [指标跃升预警] 当此日志打印出超过 100 个张量匹配时，mAP 将恢复正常水准
        logger.info(f"✅ 成功对齐并加载了 {match_count} 个预训练模块权重！")

    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)