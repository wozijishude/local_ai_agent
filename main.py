
import os
os.environ['CHROMA_TELEMETRY_ENABLED'] = '0'
os.environ['DISABLE_CHROMA_TELEMETRY'] = '1'
os.environ['CHROMA_CLIENT_TELEMETRY_ENABLED'] = '0'
import warnings
import os
import sys
import numpy as np
import io
import shutil
import hashlib
import traceback
import click
from database.chroma_db import ChromaDB
from processors.pdf_processor import PDFProcessor
from processors.image_processor import ImageProcessor
from models.text_encoder import TextEncoder
from utils.file_operations import (
    create_topic_directories,
    move_file_to_topic,
    get_all_pdfs,
    get_all_images
)
from config import PAPER_COLLECTION, IMAGE_COLLECTION, DEFAULT_TOPIC_DIR, IMAGE_DIR, VECTOR_DB_DIR

# ========== Windows中文命令行支持（精简优化版，无冗余） ==========
if sys.platform == 'win32':
    # 解决中文输出/输入乱码
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['CHROMA_TELEMETRY_ENABLED'] = '0'

# 过滤无关警告，保留核心报错
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# ✅ 原版初始化逻辑，参数完全匹配
db = ChromaDB()
text_encoder = TextEncoder()
pdf_processor = PDFProcessor(text_encoder)
image_processor = ImageProcessor()


@click.group()
def cli():
    """本地多模态AI智能文献与图像管理助手"""
    pass


@cli.command()
@click.argument('path')
@click.option('--topics', help='逗号分隔的主题列表，如"计算机视觉,自然语言处理,机器学习"')
def add_paper(path, topics):
    """添加并分类论文"""
    if not os.path.exists(path):
        click.echo(f"❌ 错误：文件或目录不存在 - {path}")
        return

    topics = [t.strip() for t in topics.split(',')] if topics else []
    if not topics:
        click.echo("⚠️ 警告：未传入主题列表，将跳过分类直接入库")
    create_topic_directories(topics)

    pdf_files = [path] if os.path.isfile(path) and path.endswith('.pdf') else get_all_pdfs(path)
    if not pdf_files:
        click.echo("ℹ️ 未找到任何PDF文件")
        return

    # 预生成所有主题的向量（仅生成1次，提升效率+保证向量空间一致）
    topic_embeddings = []
    if topics:
        topic_embeddings = pdf_processor.text_encoder.encode(topics)
        click.echo(f"✅ 已加载 {len(topics)} 个主题向量，开始智能分类...")

    for pdf_file in pdf_files:
        try:
            # 1. 获取论文一维向量
            paper_embedding = pdf_processor.get_embedding(pdf_file)
            best_topic = None
            new_path = pdf_file

            if topics and topic_embeddings:
                # 标准化余弦相似度计算（内置异常兜底）
                def cosine_similarity(vec1, vec2):
                    if not vec1 or not vec2 or len(vec1) != len(vec2):
                        return 0.0
                    vec1 = np.array(vec1)
                    vec2 = np.array(vec2)
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0.0

                # 计算论文与所有主题的相似度
                similarities = [cosine_similarity(paper_embedding, te) for te in topic_embeddings]

                # 打印相似度值（调试关键）
                click.echo(f"\n📄 论文 {os.path.basename(pdf_file)} 与各主题相似度：")
                for t, s in zip(topics, similarities):
                    click.echo(f"   📌 {t}: {s:.4f}")

                # 选中相似度最高的主题
                max_sim_idx = np.argmax(similarities)
                best_topic = topics[max_sim_idx]
                max_sim = similarities[max_sim_idx]

                # 移动文件到最优主题目录
                new_path = move_file_to_topic(pdf_file, best_topic)
                click.echo(f"✅ 最优匹配主题：{best_topic}（相似度 {max_sim:.4f}），已完成文件迁移")

            # 2. 入库向量数据 ✅ 修复ID生成：改用MD5稳定哈希，无冲突风险
            file_name = os.path.basename(new_path)
            file_id = hashlib.md5(new_path.encode('utf-8')).hexdigest()
            db.add_embeddings(
                PAPER_COLLECTION,
                embeddings=[paper_embedding],
                metadatas=[{"path": new_path, "filename": file_name, "topic": best_topic or "未分类"}],
                ids=[f"paper_{file_id}"]
            )
            click.echo(f"✅ 论文 {file_name} 向量数据已成功入库\n")

        except Exception as e:
            # ✅ 修复异常捕获：打印完整堆栈，方便定位错误
            click.echo(f"❌ 处理 {pdf_file} 时出错: {str(e)}")
            click.echo(f"📝 错误堆栈：{traceback.format_exc()}\n")

@cli.command()
@click.argument('query')
def search_paper(query):
    """语义搜索论文【纯净优化版】：适配所有检索场景+无冗余警告+精准匹配"""
    # ========== 全局根治：彻底关闭ChromaDB遥测，消灭所有capture()警告 ==========
    import os
    os.environ['CHROMA_TELEMETRY_ENABLED'] = '0'
    os.environ['DISABLE_CHROMA_TELEMETRY'] = '1'

    try:
        click.echo(f"🔍 正在语义检索：{query}")

        # ========== 安全的检索词优化（仅清洗，不追加任何无关内容！适配所有场景） ==========
        import re
        # 只做2件事：1.去除中文无意义虚词/标点 2.统一空格分隔 → 不污染检索词本意
        clean_query = re.sub(r'[？。，！、的是啥什么为何如何()]', '', query)
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()  # 合并多余空格
        click.echo(f"✨ 检索词智能清洗：{clean_query}\n")

        # ========== 生成检索向量（维度严格兜底） ==========
        query_embedding_2d = pdf_processor.text_encoder.encode(clean_query)
        query_embedding = query_embedding_2d[0] if (query_embedding_2d and len(query_embedding_2d) > 0) else [0.0] * 384

        # 强制对齐384维，杜绝维度问题
        if len(query_embedding) != 384:
            if len(query_embedding) > 384:
                query_embedding = query_embedding[:384]
            else:
                query_embedding += [0.0] * (384 - len(query_embedding))

        # ========== 动态适配n_results，彻底消灭数量不匹配警告 ==========
        coll = db.client.get_or_create_collection(PAPER_COLLECTION)
        total_papers = coll.count()  # 获取库中真实论文数量
        n_results = min(10, total_papers) if total_papers > 0 else 1  # 最多查10条，不足则查全部

        # ========== 执行精准查询 ==========
        results = db.query(
            PAPER_COLLECTION,
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # ========== 结果处理+分级美化展示 ==========
        if results['metadatas'][0] and len(results['metadatas'][0]) > 0:
            # 过滤低相似度（≥0.4）+ 整理结果
            valid_results = []
            for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
                sim_score = 1 - dist  # 距离→相似度（越大越匹配）
                if sim_score >= 0.4:  # 过滤完全无关结果
                    valid_results.append({
                        "metadata": meta,
                        "similarity": sim_score
                    })

            if not valid_results:
                click.echo("❌ 未检索到有效相关论文（相似度≥0.4）")
                return

            # 按相似度降序排序（核心：相似度越高，排名越前）
            valid_results.sort(key=lambda x: x['similarity'], reverse=True)

            # 分级展示结果（直观清晰）
            click.echo(f"✅ 【论文语义检索结果】共找到 {len(valid_results)} 篇相关论文\n")
            for i, res in enumerate(valid_results, 1):
                meta = res['metadata']
                sim = res['similarity']
                # 相似度星级标注
                star = "⭐⭐⭐" if sim >= 0.7 else ("⭐⭐" if sim >= 0.5 else "⭐")
                click.echo(f"{i}. 📄 {meta['filename']}")
                click.echo(f"   📂 存储路径: {meta['path']}")
                click.echo(f"   📌 主题分类: {meta['topic']}")
                click.echo(f"   ✅ 语义相似度: {sim:.4f} {star}\n")

        else:
            click.echo("❌ 未检索到任何论文，请先执行 add_paper 命令导入PDF")

    except Exception as e:
        click.echo(f"❌ 论文检索失败：{str(e)}")


@cli.command()
@click.argument('query')
def search_image(query):
    """以文搜图：输入文本描述，检索最相似的图片【修复版】"""
    try:
        click.echo(f"🔍 正在以文搜图：{query}")
        if not query or query.strip() == "":
            click.echo("❌ 错误：检索词不能为空！")
            return

        # 生成检索文本向量（已归一化）
        query_embedding = image_processor.get_text_embedding(query)

        # 校验图片库是否有数据
        coll = db.client.get_or_create_collection(IMAGE_COLLECTION)
        total_imgs = coll.count()
        if total_imgs == 0:
            click.echo("❌ 未检索到图片，请先执行 index-images 命令导入图片")
            return
        n_results = min(5, total_imgs)

        # 执行向量检索
        results = db.query(
            collection_name=IMAGE_COLLECTION,
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        # 处理并展示结果（核心修复：相似度计算+去重+排序）
        if results['metadatas'][0] and len(results['metadatas'][0]) > 0:
            img_results = []
            seen_files = set()  # 用于去重，记录已处理的文件名

            for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
                file_name = meta['filename']
                # ✅ 修复1：去重 → 跳过已处理的同名图片
                if file_name in seen_files:
                    continue
                seen_files.add(file_name)

                # ✅ 修复2：L2距离 → 0~1相似度（核心公式，彻底解决负数问题）
                # L2距离越小越相似 → 映射为 0~1 区间，值越大越相似
                sim_score = 1.0 / (1.0 + dist)

                img_results.append({
                    "metadata": meta,
                    "similarity": round(sim_score, 4)  # 保留4位小数，更整洁
                })

            # ✅ 修复3：按相似度降序排序 → 最相似的图片排在最前面
            img_results.sort(key=lambda x: x['similarity'], reverse=True)

            # 美化输出结果
            click.echo(f"\n✅ 【以文搜图结果】共找到 {len(img_results)} 张相似图片\n")
            for i, res in enumerate(img_results, 1):
                meta = res['metadata']
                sim = res['similarity']
                # 星级标注：贴合0~1相似度范围
                if sim >= 0.8:
                    star = "⭐⭐⭐（极高相似度）"
                elif sim >= 0.6:
                    star = "⭐⭐（较高相似度）"
                else:
                    star = "⭐（一般相似度）"

                click.echo(f"{i}. 🖼️ {meta['filename']}")
                click.echo(f"   📂 存储路径: {meta['path']}")
                click.echo(f"   ✅ 相似度: {sim:.4f} {star}\n")
        else:
            click.echo("❌ 未找到与检索词匹配的图片")

    except Exception as e:
        click.echo(f"❌ 图像检索失败：{str(e)}")

@cli.command()
@click.argument('path')
def index_images(path):
    """索引图片文件，生成特征向量入库【完整版】"""
    try:
        # 绝对路径转换，解决相对路径报错问题
        abs_img_path = os.path.abspath(path)
        if not os.path.exists(abs_img_path):
            click.echo(f"❌ 错误：图片路径不存在 → {abs_img_path}")
            return

        # 获取所有合法图片文件
        image_files = get_all_images(abs_img_path)
        if not image_files:
            click.echo("ℹ️ 未找到有效图片文件（仅支持 jpg/png/bmp 格式）")
            return

        click.echo(f"ℹ️ 成功找到 {len(image_files)} 个图像文件，开始批量索引...\n")

        # 批量处理图片入库
        success_count = 0
        for img_file in image_files:
            try:
                # 生成图片向量（格式合规：List，维度合规：384）
                img_embedding = image_processor.get_embedding(img_file)
                # 构造元数据和唯一ID
                file_name = os.path.basename(img_file)
                metadatas = [{"path": img_file, "filename": file_name, "type": "image"}]
                # ids = [f"img_{hash(img_file)}"]
                ids = [f"img_{hashlib.md5(img_file.encode('utf-8')).hexdigest()}"]

                # 向量入库
                db.add_embeddings(IMAGE_COLLECTION, [img_embedding], metadatas, ids)
                click.echo(f"✅ 成功索引 → {file_name}")
                success_count += 1
            except Exception as e:
                click.echo(f"❌ 索引失败 → {os.path.basename(img_file)} | 原因：{str(e)}")

        # 最终统计结果
        click.echo(f"\n📊 图片索引完成！成功：{success_count} | 失败：{len(image_files) - success_count}")

    except Exception as e:
        click.echo(f"\n❌ 图片索引任务整体失败：{str(e)}")

if __name__ == '__main__':
    # 启动命令行主程序
    cli()