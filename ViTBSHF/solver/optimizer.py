# solver/optimizer.py
import torch

def make_optimizer(cfg, model):
    """
    构建优化器 (Optimizer Builder)。
    核心逻辑：对 Bias 和 BatchNorm/LayerNorm 参数不进行权重衰减 (Weight Decay)，以提升泛化能力。
    """
    params = []

    bias_lr_factor = getattr(cfg.SOLVER, 'BIAS_LR_FACTOR', 1.0)
    weight_decay_bias = getattr(cfg.SOLVER, 'WEIGHT_DECAY_BIAS', 0.0)

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue

        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * bias_lr_factor
            weight_decay = weight_decay_bias

        if "norm" in key or "bn" in key:
            weight_decay = weight_decay_bias

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer_name = getattr(cfg.SOLVER, 'OPTIMIZER_NAME', 'AdamW')
    
    if optimizer_name == 'SGD':
        momentum = getattr(cfg.SOLVER, 'MOMENTUM', 0.9)
        optimizer = torch.optim.SGD(params, momentum=momentum)
    elif optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params)
    else:
        optimizer = torch.optim.AdamW(params)

    return optimizer