import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TOPIC_DIR = os.path.join(BASE_DIR, "documents")  # PDF分类目录
IMAGE_DIR = os.path.join(BASE_DIR, "images")             # 图片目录
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")      # 向量库目录

PAPER_COLLECTION = "academic_papers"
IMAGE_COLLECTION = "image_collection"