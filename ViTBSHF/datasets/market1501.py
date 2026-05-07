import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class Market1501(BaseImageDataset):
    """
    Market-1501 数据集加载器
    """
    def __init__(self, root='datasets', verbose=True, **kwargs):
        super(Market1501, self).__init__()
        
        # [核心修复] 根据你的服务器截图，bounding_box_train 等目录直接位于 yaml 配置的 ROOT_DIR 之下。
        # 因此，直接将传入的 root (即 cfg.DATASETS.ROOT_DIR) 作为 dataset_dir，无需额外拼接 'market1501' 文件夹。
        self.dataset_dir = root
        
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # 解析不同子集的图像路径与 ID
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market-1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """检查数据集路径是否存在"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"'{self.dataset_dir}' 不存在，请检查 YAML 中的 ROOT_DIR 路径。")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"找不到训练目录: '{self.train_dir}'")

    def _process_dir(self, dir_path, relabel=False):
        """
        核心解析函数:
        文件名格式: [person_id]_[camera_id]_s[sequence_id]_[frame_id]_[detection_id].jpg
        示例: 0001_c1s1_001051_03.jpg
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1: continue  # 忽略垃圾样本 junk images
            pid_container.add(pid)
        
        # 训练集需要 Relabel (从 0 开始连续编号) 以适配分类头
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1: continue  # 忽略 junk images
            assert 0 <= pid <= 1501  # Market-1501 ID 范围校验
            assert 1 <= camid <= 6   # Market-1501 摄像头 1-6
            
            # 这里的 camid 建议减 1，使其从 0 开始计数，方便跨相机对齐偏置计算
            camid -= 1 
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset