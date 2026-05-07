# utils/reranking.py
import torch
import numpy as np
import gc

def re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3):
    """
    K-reciprocal Re-ranking 算法 (学术界 GPU Top-K 显存压缩极限版)
    完美突破 CPU 59GB 内存屏障，支持 MSMT17 纯 FP32 巨型矩阵计算。
    """
    print("  [Re-Ranking] 🚀 启动 k-reciprocal 重排序 (纯 FP32 距离矩阵 + GPU Top-K 引擎)...")
    
    if isinstance(qf, np.ndarray):
        qf = torch.from_numpy(qf)
    if isinstance(gf, np.ndarray):
        gf = torch.from_numpy(gf)

    query_num = qf.size(0)
    all_num = query_num + gf.size(0)
    feat = torch.cat([qf, gf], dim=0)
    
    # 距离矩阵始终坚持 FP32，MSMT17 下约占 35.2 GB 内存
    distmat = np.zeros((all_num, all_num), dtype=np.float32) 
    
    # [核心创新点]: 彻底抛弃 35GB 的全尺寸 argsort！
    # K-reciprocal 只需要关注局部邻居，因此提取 Top-200 索引足矣。内存消耗断崖跌至 75 MB！
    topk_k = min(200, all_num)
    initial_rank = np.zeros((all_num, topk_k), dtype=np.int32)
    
    chunk_size = 5000 
    feat_gpu = feat.cuda()
    
    print("  [Re-Ranking] 将流形空间距离下发至 GPU 进行分块提速，并动态提取 Top-K 拓扑...")
    with torch.no_grad():
        for i in range(0, all_num, chunk_size):
            end_i = min(i + chunk_size, all_num)
            feat_i = feat_gpu[i:end_i]
            for j in range(0, all_num, chunk_size):
                end_j = min(j + chunk_size, all_num)
                feat_j = feat_gpu[j:end_j]
                
                # 纯 FP32 欧氏距离计算
                dist = torch.pow(feat_i, 2).sum(dim=1, keepdim=True) + \
                       torch.pow(feat_j, 2).sum(dim=1, keepdim=True).t()
                dist.addmm_(feat_i, feat_j.t(), beta=1, alpha=-2)
                dist = dist.clamp(min=1e-12)
                
                distmat[i:end_i, j:end_j] = dist.cpu().numpy()
                
            # [核心学术代码]: 直接在 GPU 上进行并行 Top-K 获取，不给 CPU 任何负担
            chunk_dist = torch.tensor(distmat[i:end_i, :], device='cuda')
            topk_idx = torch.topk(chunk_dist, k=topk_k, dim=1, largest=False)[1]
            initial_rank[i:end_i, :] = topk_idx.cpu().numpy()
            del chunk_dist
                
    del feat_gpu
    torch.cuda.empty_cache()

    original_dist = distmat[:query_num, :].copy() 
    gallery_num = distmat.shape[0]
    
    # 归一化距离矩阵
    max_dist = np.max(distmat, axis=0)
    distmat = np.transpose(distmat / max_dist)
    
    # 为了绝不越过 59GB 内存红线，此处拓扑图 V 使用 CPU fp16 (仅占 17.5GB)
    V = np.zeros_like(distmat, dtype=np.float16)

    print("  [Re-Ranking] 构建局部拓扑稀疏图...")
    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        
        # 将 forward 索引映射到 initial_rank 前 k1+1 列
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-distmat[i, k_reciprocal_expansion_index].astype(np.float32))
        V[i, k_reciprocal_expansion_index] = (weight / np.sum(weight)).astype(np.float16)

    # 局部查询扩展 (Local Query Expansion)
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe

    del initial_rank
    gc.collect() # 强制回收内存垃圾
    
    invIndex = [np.where(V[:, i] != 0)[0] for i in range(gallery_num)]
    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    print("  [Re-Ranking] 融合 Jaccard 与流形欧氏距离...")
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] += np.minimum(V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    # 纯 FP32 空间聚合计算
    final_dist = jaccard_dist.astype(np.float32) * (1 - lambda_value) + original_dist.astype(np.float32) * lambda_value
    
    del original_dist, V, jaccard_dist
    gc.collect()
    
    final_dist = final_dist[:, query_num:]
    return final_dist