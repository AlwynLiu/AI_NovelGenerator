#novel_generator/vectorstore_utils.py
# -*- coding: utf-8 -*-
"""
向量库相关操作（初始化、更新、检索、清空、文本切分等）
适配 chromadb 0.5.22 版本
"""
import os
import logging
import traceback
import jieba
import numpy as np
import re
import ssl
import requests
import warnings
from langchain_chroma import Chroma

# 禁用特定的Torch警告
warnings.filterwarnings('ignore', message='.*Torch was not compiled with flash attention.*')
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizer并行警告

from chromadb.config import Settings
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from .common import call_with_retry

def get_vectorstore_dir(filepath: str) -> str:
    """获取 vectorstore 路径"""
    return os.path.join(filepath, "vectorstore")

def clear_vector_store(filepath: str) -> bool:
    """清空 清空向量库"""
    import shutil
    store_dir = get_vectorstore_dir(filepath)
    if not os.path.exists(store_dir):
        logging.info("No vector store found to clear.")
        return False
    try:
        shutil.rmtree(store_dir)
        logging.info(f"Vector store directory '{store_dir}' removed.")
        return True
    except Exception as e:
        logging.error(f"无法删除向量库文件夹，请关闭程序后手动删除 {store_dir}。\n {str(e)}")
        traceback.print_exc()
        return False

def init_vector_store(embedding_adapter, texts, filepath: str):
    """
    在 filepath 下创建/加载一个 Chroma 向量库并插入 texts。
    如果Embedding失败，则返回 None，不中断任务。
    适配 chromadb 0.5.22 版本（使用 metadata 配置 HNSW）
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain.embeddings.base import Embeddings as LCEmbeddings

    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)
    documents = [Document(page_content=str(t)) for t in texts]

    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, texts):
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    texts=texts
                )
            def embed_query(self, query: str):
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query=query
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()

        # 1. 创建原生 ChromaDB 持久化客户端
        chroma_client = chromadb.PersistentClient(
            path=store_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # 2. 删除已有集合（如果有），确保新 HNSW 配置生效（可选）
        try:
            chroma_client.delete_collection("novel_collection")
            logging.info("已删除旧的 collection，将使用新的 HNSW 配置重建。")
        except ValueError:
            pass  # 集合不存在，忽略

        # 3. 使用原生客户端创建集合并配置 HNSW 参数（旧版 metadata 方式）
        #    注意：在 chromadb 0.5.x 中，参数名使用 hnsw:M / hnsw:construction_ef / hnsw:space
        collection = chroma_client.create_collection(
            name="novel_collection",
            metadata={
                "hnsw:M": 12,                 # M值，控制内存占用
                "hnsw:construction_ef": 50,   # 构建候选池大小
                "hnsw:space": "cosine"        # 距离度量
            }
        )

        # 4. 将 LangChain 的 Chroma 包装器挂载到该客户端
        vectorstore = Chroma(
            client=chroma_client,
            collection_name="novel_collection",
            embedding_function=chroma_embedding,
        )

        # 5. 分批添加文档（每批 50 条，降低内存峰值）
        batch_size = 50
        total = len(documents)
        for i in range(0, total, batch_size):
            batch = documents[i:i+batch_size]
            vectorstore.add_documents(batch)
            logging.info(f"已添加批次 {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")

        return vectorstore

    except Exception as e:
        logging.warning(f"Init vector store failed: {e}")
        traceback.print_exc()
        return None

def load_vector_store(embedding_adapter, filepath: str):
    """
    读取已存在的 Chroma 向量库。若不存在则自动创建一个空库。
    如果加载/创建失败，则返回 None。
    适配 chromadb 0.5.22 版本
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    from chromadb.errors import InvalidCollectionException  # 关键导入
    from langchain_chroma import Chroma
    from langchain.embeddings.base import Embeddings as LCEmbeddings

    store_dir = get_vectorstore_dir(filepath)
    os.makedirs(store_dir, exist_ok=True)

    try:
        class LCEmbeddingWrapper(LCEmbeddings):
            def embed_documents(self, texts):
                return call_with_retry(
                    func=embedding_adapter.embed_documents,
                    max_retries=3,
                    fallback_return=[],
                    texts=texts
                )
            def embed_query(self, query: str):
                res = call_with_retry(
                    func=embedding_adapter.embed_query,
                    max_retries=3,
                    fallback_return=[],
                    query=query
                )
                return res

        chroma_embedding = LCEmbeddingWrapper()

        chroma_client = chromadb.PersistentClient(
            path=store_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # 检查集合是否存在，若不存在则创建
        try:
            chroma_client.get_collection("novel_collection")
            logging.info("Collection 'novel_collection' already exists, loading existing.")
        except (ValueError, InvalidCollectionException):  # 捕获两种异常
            logging.info("Collection 'novel_collection' does not exist, creating a new one.")
            chroma_client.create_collection(
                name="novel_collection",
                metadata={
                    "hnsw:M": 12,
                    "hnsw:construction_ef": 50,
                    "hnsw:space": "cosine"
                }
            )
            logging.info("New collection created with HNSW config (M=12, construction_ef=50).")

        vectorstore = Chroma(
            client=chroma_client,
            collection_name="novel_collection",
            embedding_function=chroma_embedding,
        )
        return vectorstore

    except Exception as e:
        logging.warning(f"Failed to load or create vector store: {e}")
        traceback.print_exc()
        return None
        

def split_by_length(text: str, max_length: int = 500):
    """按照 max_length 切分文本"""
    segments = []
    start_idx = 0
    while start_idx < len(text):
        end_idx = min(start_idx + max_length, len(text))
        segment = text[start_idx:end_idx]
        segments.append(segment.strip())
        start_idx = end_idx
    return segments

def split_text_for_vectorstore(chapter_text: str, max_length: int = 500, similarity_threshold: float = 0.7):
    """
    对新的章节文本进行分段后,再用于存入向量库。
    使用 embedding 进行文本相似度计算。
    """
    if not chapter_text.strip():
        return []
    
    sentences = jieba.cut(chapter_text)
    if not sentences:
        return []
    
    # 直接按长度分段,不做相似度合并
    final_segments = []
    current_segment = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        if current_length + sentence_length > max_length:
            if current_segment:
                final_segments.append(" ".join(current_segment))
            current_segment = [sentence]
            current_length = sentence_length
        else:
            current_segment.append(sentence)
            current_length += sentence_length
    
    if current_segment:
        final_segments.append(" ".join(current_segment))
    
    return final_segments

def update_vector_store(embedding_adapter, new_chapter: str, filepath: str):
    """
    将最新章节文本插入到向量库中。
    若库不存在则初始化；若初始化/更新失败，则跳过。
    """
    from langchain_core.documents import Document

    splitted_texts = split_text_for_vectorstore(new_chapter)
    if not splitted_texts:
        logging.warning("No valid text to insert into vector store. Skipping.")
        return

    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("Vector store does not exist or failed to load. Initializing a new one for new chapter...")
        store = init_vector_store(embedding_adapter, splitted_texts, filepath)
        if not store:
            logging.warning("Init vector store failed, skip embedding.")
        else:
            logging.info("New vector store created successfully.")
        return

    # 追加模式：分批添加文档
    docs = [Document(page_content=str(t)) for t in splitted_texts]
    batch_size = 50
    total = len(docs)
    try:
        for i in range(0, total, batch_size):
            batch = docs[i:i+batch_size]
            store.add_documents(batch)
            logging.info(f"已添加批次 {i//batch_size + 1}/{(total + batch_size - 1)//batch_size}")
        logging.info("Vector store updated with the new chapter splitted segments.")
    except Exception as e:
        logging.warning(f"Failed to update vector store: {e}")
        traceback.print_exc()

def get_relevant_context_from_vector_store(embedding_adapter, query: str, filepath: str, k: int = 2) -> str:
    """
    从向量库中检索与 query 最相关的 k 条文本，拼接后返回。
    如果向量库加载/检索失败，则返回空字符串。
    最终只返回最多2000字符的检索片段。
    """
    store = load_vector_store(embedding_adapter, filepath)
    if not store:
        logging.info("No vector store found or load failed. Returning empty context.")
        return ""

    try:
        docs = store.similarity_search(query, k=k)
        if not docs:
            logging.info(f"No relevant documents found for query '{query}'. Returning empty context.")
            return ""
        combined = "\n".join([d.page_content for d in docs])
        if len(combined) > 2000:
            combined = combined[:2000]
        return combined
    except Exception as e:
        logging.warning(f"Similarity search failed: {e}")
        traceback.print_exc()
        return ""

def _get_sentence_transformer(model_name: str = 'paraphrase-MiniLM-L6-v2'):
    """获取sentence transformer模型，处理SSL问题"""
    try:
        # 设置torch环境变量
        os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
        os.environ["TORCH_CUDNN_V8_API_ENABLED"] = "0"
        
        # 禁用SSL验证
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # ... existing code ...
    except Exception as e:
        logging.error(f"Failed to load sentence transformer model: {e}")
        traceback.print_exc()
        return None