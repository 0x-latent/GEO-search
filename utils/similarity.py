"""
文本相似度计算工具。
清洗 + jieba分词 + TF-IDF + 余弦相似度。
"""
import re
import math
from collections import Counter


def clean_text(text: str) -> str:
    """清洗回答文本，去除markdown格式和多余空白"""
    # 去除markdown标记
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'\*{1,3}(.+?)\*{1,3}', r'\1', text)
    text = re.sub(r'`{1,3}(.+?)`{1,3}', r'\1', text)
    text = re.sub(r'\[(.+?)\]\(.+?\)', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.+?\)', '', text)
    text = re.sub(r'^[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+[.、)）]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'---+|===+|\*\*\*+', '', text)
    text = re.sub(r'\|', ' ', text)

    # 去除多余空白
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = text.strip()
    return text


def calc_similarity(texts: list) -> float:
    """
    计算多段文本的平均两两TF-IDF余弦相似度。

    Args:
        texts: 清洗后的文本列表

    Returns:
        0-1之间的平均相似度
    """
    import jieba

    if len(texts) < 2:
        return 1.0

    # jieba分词，过滤单字符
    tokenized = []
    for t in texts:
        words = [w for w in jieba.cut(t) if len(w) > 1]
        tokenized.append(words)

    # 文档频率
    doc_count = len(tokenized)
    df = Counter()
    for words in tokenized:
        for w in set(words):
            df[w] += 1

    # TF-IDF向量
    def tfidf_vector(words):
        tf = Counter(words)
        total = len(words) if words else 1
        vec = {}
        for w, count in tf.items():
            tf_val = count / total
            idf_val = math.log((doc_count + 1) / (df[w] + 1)) + 1
            vec[w] = tf_val * idf_val
        return vec

    vectors = [tfidf_vector(words) for words in tokenized]

    # 两两余弦相似度
    def cosine_sim(v1, v2):
        common = set(v1.keys()) & set(v2.keys())
        if not common:
            return 0.0
        dot = sum(v1[k] * v2[k] for k in common)
        norm1 = math.sqrt(sum(v ** 2 for v in v1.values()))
        norm2 = math.sqrt(sum(v ** 2 for v in v2.values()))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(cosine_sim(vectors[i], vectors[j]))

    return sum(sims) / len(sims) if sims else 1.0
