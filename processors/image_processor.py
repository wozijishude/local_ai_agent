
import os
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

class ImageProcessor:
    def __init__(self):
        """✅ 核心修复：彻底消除慢速处理器警告 + 初始化CLIP模型"""
        # 加载模型时，同时给tokenizer和processor配置use_fast=True，双保险消除警告
        self.model = SentenceTransformer(
            'clip-ViT-B-32',
            model_kwargs={"use_fast": True},
            tokenizer_kwargs={"use_fast": True}
        )
        self.FIXED_DIM = 384  # 固定384维，和文本向量对齐

    def load_image(self, img_path):
        """加载图片，统一RGB格式，兼容所有常见格式"""
        try:
            img = Image.open(img_path).convert('RGB')
            return img
        except Exception as e:
            raise Exception(f"图片加载失败：{str(e)}")

    def get_image_embedding(self, img_path):
        """图片→向量：输出List格式+384维，适配ChromaDB"""
        img = self.load_image(img_path)
        img_embedding_np = self.model.encode(img, normalize_embeddings=True)  # 归一化向量，提升匹配精度
        img_embedding = img_embedding_np.tolist()
        # 维度兜底
        if len(img_embedding) != self.FIXED_DIM:
            img_embedding = img_embedding[:self.FIXED_DIM] if len(img_embedding)>self.FIXED_DIM else img_embedding + [0.0]*(self.FIXED_DIM-len(img_embedding))
        return img_embedding

    def get_text_embedding(self, text):
        """文本→向量：输出List格式+384维，适配ChromaDB"""
        if not text or text.strip() == "":
            return [0.0] * self.FIXED_DIM
        text_embedding_np = self.model.encode(text.strip(), normalize_embeddings=True)  # 归一化向量
        text_embedding = text_embedding_np.tolist()
        # 维度兜底
        if len(text_embedding) != self.FIXED_DIM:
            text_embedding = text_embedding[:self.FIXED_DIM] if len(text_embedding)>self.FIXED_DIM else text_embedding + [0.0]*(self.FIXED_DIM-len(text_embedding))
        return text_embedding

    # 兼容旧代码
    def get_embedding(self, img_path):
        return self.get_image_embedding(img_path)