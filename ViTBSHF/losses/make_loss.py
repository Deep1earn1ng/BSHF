# losses/make_loss.py
import torch
import torch.nn.functional as F
from torch import nn

def euclidean_dist(x, y):
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist = dist - 2 * torch.matmul(x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

class TripletLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if margin > 0:
            self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        else:
            self.ranking_loss = None

    def forward(self, inputs, targets):
        dist_mat = euclidean_dist(inputs, inputs)
        n = inputs.size(0)
        is_pos = targets.expand(n, n).eq(targets.expand(n, n).t())
        is_neg = targets.expand(n, n).ne(targets.expand(n, n).t())

        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist_mat[i][is_pos[i]].max().unsqueeze(0))
            dist_an.append(dist_mat[i][is_neg[i]].min().unsqueeze(0))
            
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        if self.ranking_loss is not None:
            loss = self.ranking_loss(dist_an, dist_ap, y)
        else:
            loss = F.softplus(dist_ap - dist_an).mean()
        return loss

class PairwiseCircleLoss(nn.Module):
    def __init__(self, m=0.25, gamma=256):
        super(PairwiseCircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.soft_plus = nn.Softplus()

    def forward(self, features, targets):
        features = F.normalize(features, p=2, dim=1)
        sim_mat = torch.matmul(features, features.t())
        
        is_pos = targets.view(-1, 1).expand(targets.size(0), targets.size(0)).eq(
            targets.view(1, -1).expand(targets.size(0), targets.size(0))).float()
        is_neg = 1 - is_pos
        
        is_pos = is_pos - torch.eye(targets.size(0), device=targets.device)
        
        s_p = sim_mat * is_pos
        s_n = sim_mat * is_neg
        
        alpha_p = torch.clamp_min(-s_p.detach() + 1 + self.m, min=0.)
        alpha_n = torch.clamp_min(s_n.detach() + self.m, min=0.)
        
        delta_p = 1 - self.m
        delta_n = self.m
        
        logit_p = -alpha_p * (s_p - delta_p) * self.gamma
        logit_n = alpha_n * (s_n - delta_n) * self.gamma
        
        logit_p[is_pos == 0] = -1e9
        logit_n[is_neg == 0] = -1e9
        
        loss = self.soft_plus(torch.logsumexp(logit_p, dim=1) + torch.logsumexp(logit_n, dim=1)).mean()
        return loss

def make_loss(cfg, num_classes):
    ce_loss = CrossEntropyLabelSmooth(num_classes=num_classes)
    
    triplet_margin = getattr(cfg.MODEL, 'TRIPLET_MARGIN', 0.3)
    triplet_loss = TripletLoss(margin=triplet_margin)
    
    circle_loss = None
    if_with_circle = getattr(cfg.MODEL, 'IF_WITH_CIRCLE', 'no')
    if if_with_circle == 'yes':
        circle_loss = PairwiseCircleLoss()

    def loss_func(score, feat, target):
        loss_ce = ce_loss(score, target)
        loss_triplet = triplet_loss(feat, target)
        
        total_loss = loss_ce + loss_triplet
        
        if circle_loss is not None:
            loss_circle = circle_loss(feat, target)
            circle_weight = getattr(cfg.MODEL, 'CIRCLE_WEIGHT', 0.0005)
            total_loss += circle_weight * loss_circle

        return total_loss

    # 移除了冗余的 None 返回值
    return loss_func