# datasets/make_dataloader.py
import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

# ==============================================================================
# 显式导入自定义的数据集与采样器
# ==============================================================================
from .bases import ImageDataset
from .market1501 import Market1501
from .dukemtmc import DukeMTMCreID
from .samplers import RandomIdentitySampler
from .msmt17 import MSMT17

# 数据集注册工厂：解耦数据集加载逻辑，方便未来向 DukeMTMC 或 MSMT17 泛化
__factory = {
    
    'market': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
}

def make_dataloader(cfg):
    logger = logging.getLogger("reid.dataset")
    logger.info(">>> 🚀 正在构建数据流管道 (DataLoader)...")

    # 1. 注册并初始化数据集
    dataset_name = getattr(cfg.DATASETS, 'NAMES', 'market1501').lower()
    if dataset_name not in __factory:
        raise KeyError(f"❌ 尚未支持或注册该数据集: {dataset_name}。请检查 yaml 配置。")
    
    # 实例化数据集，自动解析 train, query, gallery 路径
    dataset = __factory[dataset_name](root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    num_cameras = dataset.num_train_cams
    num_query = len(dataset.query)

    # ==============================================================================
    # 2. 数据增强流水线 (Data Augmentation)
    # 引入 RandomErasing 模拟行人遮挡挑战，这对验证 HBF 模块的表观特征融合能力至关重要
    # ==============================================================================
    train_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        T.Pad(cfg.INPUT.PADDING),
        T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
        # 严格使用 torchvision 原生 API，彻底规避本地文件 ImportError
        T.RandomErasing(p=cfg.INPUT.RE_PROB, value=cfg.INPUT.PIXEL_MEAN)
    ])

    # 测试集必须保持原始图像比例的纯净提取
    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST, interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    # ==============================================================================
    # 3. 硬件性能榨取配置：利用 15 核 CPU 和 59GB 内存
    # ==============================================================================
    num_workers = getattr(cfg.DATALOADER, 'NUM_WORKERS', 8)

    # 4. 构建 训练集加载器 (Train Loader)
    train_set = ImageDataset(dataset.train, train_transforms)
    train_loader = DataLoader(
        train_set, 
        batch_size=cfg.SOLVER.IMS_PER_BATCH,
        # 使用随机身份采样器 (RandomIdentitySampler) 确保每个 Batch 内包含 P 个 ID，每个 ID 包含 K 张图
        # 这是支撑 Triplet Loss 进行 Hard Mining 的核心基石
        sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
        num_workers=num_workers, 
        pin_memory=True,  # 开启锁页内存，极大提升 CPU 到 GPU 的数据拷贝带宽
        drop_last=True
    )

    # ==============================================================================
    # 5. 构建 验证集加载器 (Validation Loader)
    # [核心修复] 彻底解决 YACS 配置引擎中 TEST 节点缺失引发的级联崩溃
    # ==============================================================================
    # 先判断 cfg 是否有 TEST 根节点，再判断是否有 IMS_PER_BATCH 子节点
    if hasattr(cfg, 'TEST') and hasattr(cfg.TEST, 'IMS_PER_BATCH'):
        val_batch_size = cfg.TEST.IMS_PER_BATCH
    else:
        # 在推理测试阶段不需要保存梯度，显存占用极小。
        # 既然你有 64GB 显存，我们直接将 val_batch_size 拉到 256 以实现极速推理
        val_batch_size = 256
        logger.warning(f"⚠️ 未检测到 cfg.TEST.IMS_PER_BATCH。基于 64GB 显存，自动将验证集 Batch Size 设为 {val_batch_size}")

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, 
        batch_size=val_batch_size, 
        shuffle=False,  # 验证集严格禁止 Shuffle，确保 Distance Matrix 索引映射一致
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    logger.info(f"✅ DataLoader 管道初始化完毕! 训练类别(IDs): {num_classes}, 摄像头(Cams): {num_cameras}")

    return train_loader, val_loader, num_query, num_classes, num_cameras