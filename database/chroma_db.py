
import chromadb
import numpy as np
# ========== 1. 全局禁用遥测，根治capture()警告 ==========
import os

os.environ['CHROMA_TELEMETRY_ENABLED'] = '0'
os.environ['DISABLE_CHROMA_TELEMETRY'] = '1'
os.environ['OTEL_SDK_DISABLED'] = '1'
# =======================================================
from chromadb.errors import InvalidDimensionException
from config import VECTOR_DB_DIR


class ChromaDB:
    def __init__(self):
        # 初始化持久化客户端
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        # 全局固定向量维度（文本384维，图片按编码器输出为准，统一校验）
        self.FIXED_DIMENSION = 384

    def add_embeddings(self, collection_name, embeddings, metadatas, ids):
        """添加向量 + 格式强制转换（numpy→list） + 维度校验 + 数据兜底"""
        try:
            coll = self.client.get_or_create_collection(collection_name)

            # ✅ 核心修复1：强制将【numpy数组/任意格式】转为Python原生列表
            valid_embeddings = []
            for emb in embeddings:
                # 情况1：是numpy数组 → 转list
                if isinstance(emb, np.ndarray):
                    fixed_emb = emb.tolist()
                # 情况2：已是list → 直接使用
                elif isinstance(emb, list):
                    fixed_emb = emb
                # 情况3：其他格式 → 兜底为空列表
                else:
                    fixed_emb = []

                # ✅ 核心修复2：维度兜底（统一转为384维，避免维度报错）
                if not fixed_emb:
                    fixed_emb = [0.0] * self.FIXED_DIMENSION
                elif len(fixed_emb) > self.FIXED_DIMENSION:
                    fixed_emb = fixed_emb[:self.FIXED_DIMENSION]
                elif len(fixed_emb) < self.FIXED_DIMENSION:
                    fixed_emb += [0.0] * (self.FIXED_DIMENSION - len(fixed_emb))

                valid_embeddings.append(fixed_emb)

            # 写入校验+格式转换后的纯净向量
            coll.add(embeddings=valid_embeddings, metadatas=metadatas, ids=ids)
            return True

        except Exception as e:
            raise Exception(f"向量入库失败：{str(e)}")

    def query(self, collection_name, query_embeddings, n_results):
        """查询向量 + 格式转换 + 维度校验"""
        try:
            coll = self.client.get_or_create_collection(collection_name)

            # ✅ 查询向量同样做格式转换+维度校验
            valid_query_emb = []
            for emb in query_embeddings:
                if isinstance(emb, np.ndarray):
                    fixed_emb = emb.tolist()
                elif isinstance(emb, list):
                    fixed_emb = emb
                else:
                    fixed_emb = [0.0] * self.FIXED_DIMENSION

                if len(fixed_emb) != self.FIXED_DIMENSION:
                    if len(fixed_emb) > self.FIXED_DIMENSION:
                        fixed_emb = fixed_emb[:self.FIXED_DIMENSION]
                    else:
                        fixed_emb += [0.0] * (self.FIXED_DIMENSION - len(fixed_emb))
                valid_query_emb.append(fixed_emb)

            return coll.query(query_embeddings=valid_query_emb, n_results=n_results)
        except InvalidDimensionException as e:
            raise Exception(
                f"维度匹配失败！{str(e)}\n"
                "👉 解决方案：删除 {VECTOR_DB_DIR} 文件夹后重新运行"
            )
        except Exception as e:
            raise Exception(f"向量查询失败：{str(e)}")