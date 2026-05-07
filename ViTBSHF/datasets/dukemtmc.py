import glob
import re
import os.path as osp
from .bases import BaseImageDataset

class DukeMTMCreID(BaseImageDataset):
    """
    DukeMTMC-reID 数据集加载器
    学术说明：DukeMTMC 包含 8 个摄像头，背景存在严重的非对齐和树木遮挡问题。
    """
    def __init__(self, root='/dev/shm/deepwrh_duke/', verbose=True, **kwargs):
        super(DukeMTMCreID, self).__init__()
        
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
            print("=> DukeMTMC-reID loaded")
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
            raise RuntimeError(f"'{self.dataset_dir}' 不存在，请确保已将 DukeMTMC 挂载至 /dev/shm 或正确配置 ROOT_DIR。")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"找不到训练目录: '{self.train_dir}'")

    def _process_dir(self, dir_path, relabel=False):
        """
        核心解析函数:
        文件名格式: [person_id]_c[camera_id]_f[frame_id].jpg
        示例: 0005_c2_f0046985.jpg
        """
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

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
            
            # DukeMTMC 摄像头约束校验 (1-8)
            assert 1 <= camid <= 8
            
            # 对齐偏置空间：将 CamID 转换为 0-based
            camid -= 1 
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset