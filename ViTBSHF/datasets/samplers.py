# datasets/samplers.py
import copy
import random
import numpy as np
from collections import defaultdict
from torch.utils.data.sampler import Sampler

class RandomIdentitySampler(Sampler):
    """
    随机身份采样器 (Random Identity Sampler)
    学术依据：在 ReID 任务中，为了计算 Triplet Loss 的 Hard Mining，
    必须确保每个 Batch 内包含 P 个身份 (PID)，每个身份包含 K 张图片。
    """
    def __init__(self, data_source, batch_size, num_instances):
        # 严格适配 PyTorch 2.0+ 架构规范，移除废弃的 data_source 传参
        super(RandomIdentitySampler, self).__init__()
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        self.index_dic = defaultdict(list)
        for index, (_, pid, _) in enumerate(self.data_source):
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                # 若某ID图片不足 K 张，使用有放回重采样填补
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
                    
        return iter(final_idxs)

    def __len__(self):
        return len(self.data_source)