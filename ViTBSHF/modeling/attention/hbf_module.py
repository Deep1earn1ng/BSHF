import torch
import torch.nn as nn

class HBFModule(nn.Module):
    """
    Hierarchical Background Suppression and Feature Fusion (HBF)
    学术创新点：可微显著性加权与前背景解耦特征融合 (包含边界安全约束)
    """
    def __init__(self, embed_dim=768, keep_ratio=0.7):
        super(HBFModule, self).__init__()
        
        self.embed_dim = embed_dim
        self.keep_ratio = keep_ratio
        
        # 显著性探测器
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4, bias=False),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 1, bias=False)
        )

    def forward(self, x):
        cls_token = x[:, 0:1, :]
        patch_tokens = x[:, 1:, :]
        B, N, D = patch_tokens.shape
        
        # 1. 可微的显著性评分 
        raw_scores = self.scorer(patch_tokens) 
        scores_prob = torch.sigmoid(raw_scores) 
        
        # 将分数乘回特征图，建立完整的计算图反向传播路径
        weighted_tokens = patch_tokens * scores_prob
        
        # 2. 动态截断点计算
        K = int(N * self.keep_ratio)
        
        # =====================================================================
        # [学术防坑修复]: 边界安全约束 (Safety Constraint)
        # 无论动态比例如何变化，强制确保前景(K)和背景(N-K)至少各包含 1 个 Token
        # 彻底杜绝 0 维张量引发的 mean() -> NaN 的数学崩溃灾难
        # =====================================================================
        K = max(1, min(K, N - 1))
        
        sorted_scores, indices = torch.sort(scores_prob.squeeze(-1), dim=-1, descending=True)
        
        # 3. 提取高优前景与背景
        fg_indices = indices[:, :K].unsqueeze(-1).expand(-1, -1, D)
        fg_tokens = torch.gather(weighted_tokens, 1, fg_indices)
        
        # 4. 特征池化与融合
        fg_sink = fg_tokens.mean(dim=1, keepdim=True)
        
        # 将净化后的高优前景特征残差注入全局 CLS Token
        fused_global = cls_token.squeeze(1) + fg_sink.squeeze(1)
        
        if self.training:
            # 此时 K 绝对大于 0 且小于 N，mean() 绝对安全
            fg_scores_avg = sorted_scores[:, :K].mean(dim=1)
            bg_scores_avg = sorted_scores[:, K:].mean(dim=1)
            return fused_global, fg_scores_avg, bg_scores_avg
            
        return fused_global