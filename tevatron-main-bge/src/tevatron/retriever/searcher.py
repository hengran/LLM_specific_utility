import faiss
import numpy as np
import os
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

# class FaissFlatSearcher:
#     def __init__(self, init_reps: np.ndarray):
#         # 打印环境变量用于调试
#         print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}")
        
#         self.dim = init_reps.shape[1]
#         self.index = None
        
#         # 获取GPU数量
#         ngpu = faiss.get_num_gpus()
#         print(f"检测到的GPU数量: {ngpu}")
        
#         if ngpu > 0:
#             # 方法1: 使用推荐的index_cpu_to_all_gpus (不传递gpu_resources)
#             try:
#                 co = faiss.GpuMultipleClonerOptions()
#                 co.shard = True  # 数据分片到各GPU
#                 self.index = faiss.index_cpu_to_all_gpus(
#                     faiss.IndexFlatIP(self.dim),
#                     co=co
#                 )
#                 print("成功创建分布式GPU索引")
#             except Exception as e1:
#                 # 方法1失败时回退到方法2
#                 print(f"分布式GPU索引创建失败: {str(e1)}")
#                 print("尝试单GPU方案...")
                
#                 # 创建单个GPU索引 (使用第一个可见GPU)
#                 gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
#                 res = faiss.StandardGpuResources()
#                 self.index = faiss.index_cpu_to_gpu(
#                     res,
#                     gpu_id,
#                     faiss.IndexFlatIP(self.dim)
#                 )
#                 print(f"使用单GPU (ID: {gpu_id})")
#         else:
#             # 回退到CPU
#             print("警告: 使用CPU索引")
#             self.index = faiss.IndexFlatIP(self.dim)
        
#         # 添加初始化数据
#         self.index.add(init_reps)
#         print(f"索引已初始化, 包含 {self.index.ntotal} 个向量")
class FaissFlatSearcher:
    def __init__(self, init_reps: np.ndarray):
        index = faiss.IndexFlatIP(init_reps.shape[1])
        # if faiss.get_num_gpus() > 0:
        #     logger.info("Using GPU")
        #     co = faiss.GpuMultipleClonerOptions()
        #     co.shard = False  # 改为复制模式，允许分批添加
        #     index = faiss.index_cpu_to_all_gpus(index, co=co)
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)  # 现在支持多次调用

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int, quiet: bool=False):
        num_query = q_reps.shape[0]
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), disable=quiet):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)
        all_indices = np.concatenate(all_indices, axis=0)

        return all_scores, all_indices


class FaissSearcher(FaissFlatSearcher):

    def __init__(self, init_reps: np.ndarray, factory_str: str):
        index = faiss.index_factory(init_reps.shape[1], factory_str)
        self.index = index
        self.index.verbose = True
        if not self.index.is_trained:
            self.index.train(init_reps)
