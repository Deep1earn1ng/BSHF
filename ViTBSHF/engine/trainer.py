# engine/trainer.py
import logging
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast  # 导入 AMP 混合精度模块
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from utils.checkpoint import save_checkpoint

import warnings
warnings.filterwarnings("ignore")

# [核心护盾]: 强制 AMD/NVIDIA 显卡在底层算子上规避死锁，但这不影响上层使用 AMP 提速！
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

def do_train(cfg, model, train_loader, val_loader, optimizer, scheduler, loss_fn, num_query, 
             start_epoch=0, **kwargs):
    
    logger = logging.getLogger("reid.train")
    logger.info(">>> 🚀 启动基于 HBF-ViT 的训练引擎 (AMP 极速混合精度 + 免疫死锁模式)...")

    device = torch.device(cfg.MODEL.DEVICE)
    epochs = cfg.SOLVER.MAX_EPOCHS
    dataset_name = getattr(cfg.DATASETS, 'NAMES', 'market1501').lower()
    
    # [性能核心]: 恢复梯度缩放器，开启硬件级加速
    scaler = GradScaler('cuda')

    best_mAP = 0.0
    best_rank1 = 0.0
    target_keep_ratio = 0.7 

    for epoch in range(start_epoch + 1, epochs + 1):
        # --- 1. 动态 HBF Keep Ratio 更新 ---
        if epoch <= 30:
            current_ratio = 1.0
        else:
            progress = (epoch - 30) / (epochs - 30)
            current_ratio = 1.0 - progress * (1.0 - target_keep_ratio)
            
        if hasattr(model, 'module'):
            model.module.hbf_module.keep_ratio = current_ratio
        else:
            model.hbf_module.keep_ratio = current_ratio
            
        logger.info(f"Epoch [{epoch}/{epochs}]: Dynamic HBF Keep Ratio adjusted to {current_ratio:.4f}")

        # --- 2. 训练循环 ---
        model.train()
        loss_meter = AverageMeter()
        start_time = time.time()

        for n_iter, batch_data in enumerate(train_loader):
            optimizer.zero_grad()
            
            img = batch_data[0].to(device)
            target = batch_data[1].to(device)
            target_cam = batch_data[2].to(device)
            
            # [性能核心]: 在前向传播开启 autocast (FP16)，速度翻倍！
            with autocast('cuda', enabled=True):
                outputs = model(img, target, cam_label=target_cam, view_label=None)
                
                if isinstance(outputs, (tuple, list)):
                    score = outputs[0]
                    feat = outputs[1]
                else:
                    score, feat = outputs, outputs

                loss_ce_triplet = loss_fn(score, feat, target)
                
                if isinstance(outputs, (tuple, list)) and len(outputs) >= 4:
                    fg_score = outputs[2]
                    bg_score = outputs[3]
                    loss_hbf = torch.clamp(bg_score - fg_score + 0.15, min=0.0).mean()
                    loss = loss_ce_triplet + 0.1 * loss_hbf
                else:
                    loss = loss_ce_triplet

            # [性能核心]: 使用 scaler 进行安全的 FP16 梯度缩放与反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            
            log_period = getattr(cfg.SOLVER, 'LOG_PERIOD', 50)
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    f"Epoch[{epoch}/{epochs}] Iter[{n_iter + 1}/{len(train_loader)}] "
                    f"Loss: {loss_meter.avg:.4f} | Base Lr: {scheduler.get_last_lr()[0]:.2e}"
                )

        scheduler.step()
        epoch_time = time.time() - start_time
        logger.info(f"==> Epoch {epoch} 训练完成, 耗时 {epoch_time:.2f}s")

        # [安全机制]: 每轮无条件存档
        logger.info(f"💾 [安全存档] 正在保存 Epoch {epoch} 的最新状态至 latest.pth ...")
        save_checkpoint(
            cfg=cfg, epoch=epoch, model=model, optimizer=optimizer,
            scheduler=scheduler, is_best_map=False, is_best_rank1=False
        )

        # --- 3. 评估循环 ---
        eval_period = getattr(cfg.SOLVER, 'EVAL_PERIOD', 10)
        if epoch % eval_period == 0 or epoch == epochs:
            logger.info(">>> 🔍 开始在测试集上进行特征提取与评估...")
            model.eval()

            # ==============================================================
            # [修改后] 直接读取 YAML 配置，不再针对 MSMT17 做屏蔽策略
            # 强制每个评估 Epoch 均执行 Re-Ranking
            # ==============================================================
            current_reranking = cfg.TEST.RE_RANKING
            if current_reranking:
                logger.info("💡 [策略] 当前 Epoch 已强制开启 k-reciprocal Re-Ranking")

            evaluator = R1_mAP_eval(
                num_query=num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM,
                reranking=current_reranking, 
                reranking_k1=cfg.TEST.RE_RANKING_K1, 
                reranking_k2=cfg.TEST.RE_RANKING_K2
            )
            evaluator.reset()

            # [稳定性核心]: 评估期也穿上 autocast 护甲，彻底斩断 AMD 显卡的假死路径！
            with torch.no_grad():
                with autocast('cuda', enabled=True): 
                    logger.info(f"  -> 🚀 开始前向传播，共计 {len(val_loader)} 个 Batches...")
                    for i, val_batch in enumerate(val_loader):
                        val_img = val_batch[0].to(device)
                        val_vid = val_batch[1]
                        val_camid = val_batch[2]
                        
                        if torch.is_tensor(val_camid):
                            val_camid_gpu = val_camid.to(device)
                        else:
                            val_camid_gpu = torch.tensor(val_camid).to(device)
                        
                        val_outputs = model(val_img, cam_label=val_camid_gpu, view_label=None)
                        
                        if isinstance(val_outputs, (tuple, list)):
                            val_feat = val_outputs[1] if len(val_outputs) > 1 else val_outputs[0]
                        else:
                            val_feat = val_outputs

                        # 强制将 FP16 特征转回 FP32 存入列表
                        evaluator.update((val_feat.float(), val_vid, val_camid))
                        
                        # [新增]: 阶段性心跳日志，打破信息黑洞
                        if (i + 1) % 50 == 0 or (i + 1) == len(val_loader):
                            logger.info(f"  -> 特征提取进度: [{i + 1}/{len(val_loader)}] Batches")

            cmc, mAP, mINP = evaluator.compute()
            
            logger.info("================ 测试结果 ================")
            logger.info(f"mAP: {mAP:.1%}")
            logger.info(f"CMC Rank-1: {cmc[0]:.1%}")
            logger.info(f"CMC Rank-5: {cmc[4]:.1%}")
            logger.info("==========================================")

            is_best_map = (mAP > best_mAP)
            is_best_rank1 = (cmc[0] > best_rank1)
            
            if is_best_map: best_mAP = mAP
            if is_best_rank1: best_rank1 = cmc[0]

            if is_best_map or is_best_rank1:
                save_checkpoint(
                    cfg=cfg, epoch=epoch, model=model, optimizer=optimizer,
                    scheduler=scheduler, is_best_map=is_best_map, is_best_rank1=is_best_rank1
                )