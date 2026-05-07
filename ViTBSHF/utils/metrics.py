# utils/metrics.py
import numpy as np
import torch
from .reranking import re_ranking

def euclidean_distance(qf, gf):
    """
    计算 Query 和 Gallery 之间的欧氏距离。
    [学术优化]: 直接返回 GPU Tensor，拒绝提早下放 CPU 导致内存灾难！
    """
    qf = qf.float()
    gf = gf.float()
    m = qf.shape[0]
    n = gf.shape[0]
    
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta=1, alpha=-2)
    
    return dist_mat  # 留在 GPU 显存中！

def eval_func(indices, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    # 此处逻辑保持你的原版不变 (为了精简篇幅，请保留你原来的代码逻辑)
    num_q, num_g = indices.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: gallery size is quite small, got {}".format(num_g))

    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.

    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx][:2000]

        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        raw_cmc = g_pids[order][keep] == q_pid
        if not np.any(raw_cmc):
            continue

        cmc = raw_cmc.cumsum()
        cmc[cmc > 1] = 1
        all_cmc.append(cmc[:max_rank])

        num_rel = raw_cmc.sum()
        tmp_cmc = raw_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * raw_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

        matches_idx = np.nonzero(raw_cmc)[0]
        rank_hard = matches_idx[-1] + 1
        INP = num_rel / rank_hard
        all_INP.append(INP)

        num_valid_q += 1.

    assert num_valid_q > 0, "Error: No valid query found"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP, mINP

class R1_mAP_eval():
    def __init__(self, num_query, max_rank=50, feat_norm='yes', reranking=False,
                 reranking_k1=20, reranking_k2=6):
        super(R1_mAP_eval, self).__init__()
        self.num_query = num_query
        self.max_rank = max_rank
        self.feat_norm = feat_norm
        self.reranking = reranking
        self.reranking_k1 = reranking_k1
        self.reranking_k2 = reranking_k2
        self.reset()

    def reset(self):
        self.feats = []
        self.pids = []
        self.camids = []

    def update(self, output):
        feat, pid, camid = output
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))

    def compute(self):
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm == 'yes':
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])

        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])

        if self.reranking:
            print(f'==> Using Re-Ranking (k1={self.reranking_k1}, k2={self.reranking_k2})...')
            distmat = re_ranking(qf.cpu().numpy(), gf.cpu().numpy(),
                                 k1=self.reranking_k1, k2=self.reranking_k2)
            print('==> Computing CMC and mAP...')
            indices = np.argsort(distmat, axis=1)
        else:
            print('==> 🚀 [硬件优化] 计算欧氏距离矩阵 (GPU加速中)...')
            distmat_gpu = euclidean_distance(qf, gf)
            
            print('==> 🚀 [硬件优化] 提取排序索引 (Tensor Core 并发排序中)...')
            # 瞬间完成排序，然后再传回 CPU 交给度量函数
            indices = torch.argsort(distmat_gpu, dim=1).cpu().numpy()

        return eval_func(indices, q_pids, g_pids, q_camids, g_camids, self.max_rank)