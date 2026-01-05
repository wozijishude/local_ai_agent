
import os
import shutil
from config import DEFAULT_TOPIC_DIR

def create_topic_directories(topics):
    """创建主题目录，不存在则新建"""
    if not os.path.exists(DEFAULT_TOPIC_DIR):
        os.makedirs(DEFAULT_TOPIC_DIR, exist_ok=True)
    for topic in topics:
        topic_dir = os.path.join(DEFAULT_TOPIC_DIR, topic)
        if not os.path.exists(topic_dir):
            os.makedirs(topic_dir, exist_ok=True)


def move_file_to_topic(file_path, topic):
    """强制复制PDF+去重，同名文件自动覆盖，避免冗余"""
    target_topic_dir = os.path.join(DEFAULT_TOPIC_DIR, topic)
    os.makedirs(target_topic_dir, exist_ok=True)  # 确保目录存在
    file_name = os.path.basename(file_path)
    target_path = os.path.join(target_topic_dir, file_name)

    # 去重：目标文件已存在则先删除，再复制
    if os.path.exists(target_path):
        os.remove(target_path)
    shutil.copy2(file_path, target_path)
    return target_path

def get_all_pdfs(folder_path):
    """遍历目录下所有PDF"""
    pdf_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
    return pdf_files

def get_all_images(folder_path):
    """遍历目录下所有图片"""
    img_suffix = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(suffix) for suffix in img_suffix):
                img_files.append(os.path.join(root, file))
    return img_files