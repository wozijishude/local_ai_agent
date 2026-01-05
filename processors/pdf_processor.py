
from PyPDF2 import PdfReader
from models.text_encoder import TextEncoder

class PDFProcessor:
    def __init__(self, text_encoder):
        self.text_encoder = text_encoder
        # 固定向量维度
        self.FIXED_DIM = 384

    def extract_core_content(self, pdf_path):
        """✅ 核心优化：只提取PDF【标题+摘要+前3页】，抛弃全文冗余内容"""
        reader = PdfReader(pdf_path)
        core_text = ""
        total_pages = len(reader.pages)

        # 1. 优先提取第一页（论文标题+摘要+引言，90%核心信息都在这）
        if total_pages >= 1:
            page1 = reader.pages[0].extract_text()
            if page1: core_text += f"【标题+摘要】{page1}\n\n"

        # 2. 补充提取第2、3页（核心架构/方法章节）
        for i in range(1, min(3, total_pages)):
            page_text = reader.pages[i].extract_text()
            if page_text: core_text += f"【核心章节{i + 1}】{page_text}\n\n"

        # 兜底：无文本时用文件名补充（增强匹配度）
        if not core_text:
            core_text = f"论文文件：{pdf_path.split('/')[-1]}"
        return core_text.strip()

    def get_embedding(self, pdf_path):
        """✅ 基于核心内容生成向量，特征更精准"""
        # 提取核心文本（标题+摘要+前3页）
        core_pdf_text = self.extract_core_content(pdf_path)
        # 生成向量
        embeddings_2d = self.text_encoder.encode(core_pdf_text)
        # 二维转一维 + 维度兜底
        embedding_1d = embeddings_2d[0] if (embeddings_2d and len(embeddings_2d) > 0) else [0.0] * self.FIXED_DIM

        # 强制对齐384维
        if len(embedding_1d) > self.FIXED_DIM:
            embedding_1d = embedding_1d[:self.FIXED_DIM]
        elif len(embedding_1d) < self.FIXED_DIM:
            embedding_1d += [0.0] * (self.FIXED_DIM - len(embedding_1d))
        return embedding_1d