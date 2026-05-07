import torch
import math
from bisect import bisect_right

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_epochs=10,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones 必须是升序排列的, 例如 [40, 70]"
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "仅支持 'constant' 或 'linear' 预热方式"
            )

        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            max_epochs,
            warmup_epochs=10,
            warmup_factor=1.0 / 3,
            eta_min=1e-6,
            last_epoch=-1,
    ):
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self.eta_min = eta_min
        super(WarmupCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = float(self.last_epoch) / self.warmup_epochs
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            progress = float(self.last_epoch - self.warmup_epochs) / \
                       max(1, (self.max_epochs - self.warmup_epochs))
            progress = min(1.0, max(0.0, progress))
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            return [
                self.eta_min + (base_lr - self.eta_min) * cosine_decay
                for base_lr in self.base_lrs
            ]


def make_scheduler(cfg, optimizer):
    max_epochs = getattr(cfg.SOLVER, 'MAX_EPOCHS', 120)
    warmup_epochs = getattr(cfg.SOLVER, 'WARMUP_EPOCHS', 10)
    warmup_factor = getattr(cfg.SOLVER, 'WARMUP_FACTOR', 0.01)
    steps = getattr(cfg.SOLVER, 'STEPS', [40, 70])
    gamma = getattr(cfg.SOLVER, 'GAMMA', 0.1)
    warmup_method = getattr(cfg.SOLVER, 'WARMUP_METHOD', "linear")
    
    scheduler_name = getattr(cfg.SOLVER, 'SCHEDULER_NAME', 'cosine')

    if scheduler_name == 'cosine':
        return WarmupCosineAnnealingLR(
            optimizer,
            max_epochs=max_epochs,
            warmup_epochs=warmup_epochs,
            warmup_factor=warmup_factor,
            eta_min=1e-6
        )
    else:
        return WarmupMultiStepLR(
            optimizer,
            milestones=steps,
            gamma=gamma,
            warmup_factor=warmup_factor,
            warmup_epochs=warmup_epochs,
            warmup_method=warmup_method
        )