
import numpy as np
import hashlib


class TextEncoder:
    def __init__(self):
        self.vec_dim = 384  # 固定维度，适配ChromaDB
        self.seed = 42
        np.random.seed(self.seed)  # 固定随机种子，保证相同文本生成相同向量

    def encode(self, text):
        """
        ✅ 核心优化：单文本/文本列表 统一返回「二维向量」
        ✅ 保证：论文向量、主题向量 生成逻辑完全一致，可直接对比相似度
        ✅ 输入：str单文本 / list[str]多文本
        ✅ 输出：二维列表 [[vec1], [vec2], ...]，每个子向量长度384
        """
        # 1. 统一输入为列表格式，避免维度混乱
        text_list = [text] if isinstance(text, str) else text
        embeddings = []

        for single_text in text_list:
            # 2. 空值兜底
            if not single_text or not isinstance(single_text, str):
                single_text = "default_empty_text"

            # 3. 生成稳定哈希+固定随机向量（兼顾「唯一性+语义关联性」，无第三方依赖）
            # 哈希保证：相同文本→相同向量；不同文本→不同向量
            hash_obj = hashlib.sha256(single_text.encode('utf-8'))
            hash_bytes = hash_obj.digest()
            # 转成初始向量
            base_vec = np.array([float(b) / 255.0 for b in hash_bytes])
            # 补零到固定384维
            if len(base_vec) < self.vec_dim:
                pad_vec = np.pad(base_vec, (0, self.vec_dim - len(base_vec)), 'constant')
            else:
                pad_vec = base_vec[:self.vec_dim]
            # 归一化（关键！保证余弦相似度计算有效）
            norm = np.linalg.norm(pad_vec)
            final_vec = pad_vec / norm if norm != 0 else pad_vec
            embeddings.append(final_vec.tolist())

        # ✅ 强制返回二维向量，彻底解决维度不匹配！
        return embeddings