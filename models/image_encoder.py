
from sentence_transformers import SentenceTransformer

class ImageEncoder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.FIXED_DIM = 384

    def encode_image(self, img_path):
        """✅ 图像文件名语义编码（和PDF逻辑一致，保证跨模态匹配）"""
        img_name = img_path.split('/')[-1].replace('.jpg','').replace('.png','').strip()
        emb = self.model.encode(img_name, normalize_embeddings=True)
        return emb[:384] if len(emb)>384 else emb

    def encode(self, text):
        """✅ 文本语义编码，和TextEncoder统一"""
        emb = self.model.encode(text, normalize_embeddings=True)
        return emb[:384] if len(emb)>384 else emb