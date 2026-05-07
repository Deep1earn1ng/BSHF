# datasets/msmt17.py
import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class MSMT17(BaseImageDataset):
    """
    MSMT17 数据集加载器
    学术说明：MSMT17 包含 15 个摄像头，跨越不同时间段和光照条件。
    极其适合验证 HBF 模块在剧烈环境噪声下的鲁棒性。
    """
    def __init__(self, root='/public/home/deepwrh/ViTBSHF/data/MSMT', verbose=True, **kwargs):
        super(MSMT17, self).__init__()
        
        self.dataset_dir = root
        
        # 严格对应您提供的目录树
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()

        # 解析不同子集的图像路径与 ID
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> MSMT17 loaded")
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
        MSMT17 文件名格式: [person_id]_c[camera_id]_[frame_id].jpg
        示例: 0001_c1_0002.jpg 或 0001_c15_0002.jpg
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        # 注意：MSMT的摄像头编号可达两位数，正则必须用 \d+
        pattern = re.compile(r'([-\d]+)_c(\d+)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1: continue  # 忽略垃圾样本
            pid_container.add(pid)
        
        # 训练集需要 Relabel (从 0 开始连续编号) 以适配线性分类头
        pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(osp.basename(img_path)).groups())
            if pid == -1: continue 
            
            # MSMT17 摄像头约束校验 (1-15)
            assert 1 <= camid <= 15
            
            # 对齐偏置空间：将 CamID 转换为 0-based
            camid -= 1 
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset