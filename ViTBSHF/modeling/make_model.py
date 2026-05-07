import torch
import torch.nn as nn
from .backbones.vit import ReID_ViT
from .attention.hbf_module import HBFModule

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class ReIDNet(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(ReIDNet, self).__init__()
        
        self.in_planes = getattr(cfg.MODEL, 'FEAT_DIM', 768)
        drop_path_rate = getattr(cfg.MODEL, 'DROP_PATH', 0.1)
        stride_size = getattr(cfg.MODEL, 'STRIDE_SIZE', [16, 16])
        
        # 依据配置决定是否分配 Camera 数量给骨干网络
        sie_camera = getattr(cfg.MODEL, 'SIE_CAMERA', False)
        
        self.base = ReID_ViT(
            img_size=cfg.INPUT.SIZE_TRAIN, 
            stride_size=stride_size, 
            drop_path_rate=drop_path_rate,
            camera=camera_num if sie_camera else 0
        )
        
        keep_ratio = getattr(cfg.MODEL, 'KEEP_RATIO', 0.7)
        self.hbf_module = HBFModule(embed_dim=self.in_planes, keep_ratio=keep_ratio)
        
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        
        self.classifier = nn.Linear(self.in_planes, num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x, label=None, cam_label=None, view_label=None):
        # 激活相机对齐
        tokens = self.base.forward_features(x, camera_id=cam_label)
        
        if self.training:
            # 接收正确的特征解包
            global_feat, fg_score, bg_score = self.hbf_module(tokens)
        else:
            global_feat = self.hbf_module(tokens)
            
        bn_feat = self.bottleneck(global_feat)
        
        if self.training:
            cls_score = self.classifier(bn_feat)
            return cls_score, global_feat, fg_score, bg_score
        else:
            return bn_feat

def make_model(cfg, num_classes, camera_num=0, view_num=0):
    return ReIDNet(num_classes=num_classes, camera_num=camera_num, view_num=view_num, cfg=cfg)