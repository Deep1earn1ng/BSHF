# tools/train.py
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import glob
import torch

from utils.logger import setup_logger
from utils.iotools import load_config, save_config

from datasets.make_dataloader import make_dataloader
from modeling.make_model import make_model
from losses.make_loss import make_loss
from solver.optimizer import make_optimizer
from solver.scheduler import make_scheduler
from engine.trainer import do_train
from utils.checkpoint import load_checkpoint, load_pretrain_vit

def get_available_configs(config_dir="configs"):
    if not os.path.exists(config_dir):
        return []
    return glob.glob(os.path.join(config_dir, "*.yml"))

def train():
    parser = argparse.ArgumentParser(description="HBF-ReID Training Pipeline")
    parser.add_argument("--config_file", default="configs/market_hbf_vit.yml", type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        potential_path = os.path.join("configs", os.path.basename(args.config_file))
        if os.path.exists(potential_path):
            args.config_file = potential_path
        else:
            raise FileNotFoundError(f"\n❌ 找不到配置文件: {args.config_file}\n")

    cfg = load_config(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    save_config(cfg, cfg.OUTPUT_DIR)
    logger = setup_logger("reid", cfg.OUTPUT_DIR)
    logger.info(f"🚀 >>> 成功加载配置: {args.config_file}")

    seed = getattr(cfg.SOLVER, 'SEED', 1234)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(cfg.MODEL.DEVICE if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, num_query, num_classes, num_cameras = make_dataloader(cfg)
    logger.info(f"📦 数据加载成功: {num_classes} IDs, {num_cameras} Cameras, Query: {num_query}")

    model = make_model(cfg, num_classes=num_classes, camera_num=num_cameras, view_num=0)
    model.to(device)

    # 移除了 center_criterion
    loss_func = make_loss(cfg, num_classes=num_classes)
    
    # 移除了 optimizer_center
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)

    start_epoch = 0
    if args.resume:
        logger.info(f"🔄 >>> 断点续训: {args.resume}")
        start_epoch = load_checkpoint(args.resume, model, optimizer, scheduler)
    else:
        logger.info("🌱 >>> 全新训练，加载预训练特征...")
        if hasattr(cfg.MODEL, 'PRETRAIN_PATH') and cfg.MODEL.PRETRAIN_PATH:
            load_pretrain_vit(model, cfg.MODEL.PRETRAIN_PATH)

    logger.info("🔥 >>> 引擎点火，进入自动混合精度 (AMP) 训练循环...")
    do_train(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_func,
        num_query=num_query,
        start_epoch=start_epoch
    )

if __name__ == "__main__":
    train()