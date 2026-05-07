# modeling/backbones/vit.py
import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer

class IBN(nn.Module):
    """
    ICS 的核心组件：Instance-Batch Normalization (IBN) 层
    有效提升跨域 (Cross-Domain) 场景下的光照和背景不变性
    """
    def __init__(self, planes):
        super(IBN, self).__init__()
        half1 = int(planes / 2)
        self.half = half1
        self.IN = nn.InstanceNorm2d(half1, affine=True)
        self.BN = nn.BatchNorm2d(planes - half1)

    def forward(self, x):
        split = torch.split(x, self.half, 1)
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())
        out = torch.cat((out1, out2), 1)
        return out

class ICS_Stem(nn.Module):
    """
    TransReID 提出的 IBN-based Convolution Stem (ICS)
    完美替代 ViT 原始的单个大 Kernel 线性投影 (PatchEmbed)
    """
    def __init__(self, in_chans=3, embed_dim=768, stride_size=(16, 16)):
        super().__init__()
        
        # 为了保证总下采样率等于 stride_size，前两层卷积固定 stride=2 (即 2*2=4)
        # 最后一层卷积的 stride 动态适配，通常 stride_size 为 16 时，这里是 4
        stride_last_h = stride_size[0] // 4
        stride_last_w = stride_size[1] // 4
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            IBN(128),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=(stride_last_h, stride_last_w), padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        # 将 [B, C, H, W] 展平并转置为 [B, N, C]
        return x.flatten(2).transpose(1, 2)

class ReID_ViT(VisionTransformer):
    """
    针对 ReID 定制的 Vision Transformer，融合 SIE (Side Information Embedding) 与 ICS
    """
    def __init__(self, img_size=(384, 128), patch_size=16, stride_size=16, in_chans=3, embed_dim=768, 
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, camera=0, view=0, using_ics=True):
        
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=0, 
                         embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                         qkv_bias=qkv_bias, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, 
                         drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        
        if isinstance(stride_size, (list, tuple)):
            stride_h, stride_w = stride_size[0], stride_size[1]
        else:
            stride_h, stride_w = stride_size, stride_size
            
        if isinstance(patch_size, (list, tuple)):
            patch_h, patch_w = patch_size[0], patch_size[1]
        else:
            patch_h, patch_w = patch_size, patch_size

        self.using_ics = using_ics

        if self.using_ics:
            # [核心替换]: 使用 ICS Stem 完全覆盖 timm 默认的 patch_embed
            self.patch_embed = ICS_Stem(in_chans=in_chans, embed_dim=embed_dim, stride_size=(stride_h, stride_w))
            
            # 重新计算 Sequence Length，以确保 pos_embed 初始化维度正确
            grid_size = (img_size[0] // stride_h, img_size[1] // stride_w)
            num_patches = grid_size[0] * grid_size[1]
            self.patch_embed.num_patches = num_patches
            
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=.02)
        else:
            # 兼容模式：如果不使用 ICS，则保留你原来的 PatchEmbed 重定义逻辑
            if stride_h != patch_h or stride_w != patch_w:
                self.patch_embed.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(stride_h, stride_w))
                grid_size = (img_size[0] // stride_h, img_size[1] // stride_w)
                num_patches = grid_size[0] * grid_size[1]
                self.patch_embed.num_patches = num_patches
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
                nn.init.trunc_normal_(self.pos_embed, std=.02)

        # [学术重构]: 引入相机与视角先验嵌入 (Camera/View Embeddings)
        self.camera = camera
        if self.camera > 0:
            self.cam_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            nn.init.trunc_normal_(self.cam_embed, std=.02)

        self.in_planes = embed_dim

    def forward_features(self, x, camera_id=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # 注入跨镜感知信息
        if self.camera > 0 and camera_id is not None:
            # 广播机制: [B] 的 camera_id 索引出 [B, 1, D] 的嵌入，自动叠加到 [B, N+1, D] 上
            x = x + self.cam_embed[camera_id]

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x