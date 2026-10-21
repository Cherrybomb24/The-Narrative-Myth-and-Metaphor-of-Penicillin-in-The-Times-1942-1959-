
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""词向量相似度查询工具
功能：输入一个词，输出与该词余弦相似度最高的前50个词
"""
import numpy as np
import sys
import os


def load_word_vectors(filepath):
    """加载Word2Vec格式的词向量文件

    文件格式：
    第一行：词表大小 向量维度
    后续每行：词 向量值1 向量值2 ... 向量值N
    """
    word_vectors = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        # 读取第一行（词表大小和维度）
        header = f.readline().strip().split()
        vocab_size = int(header[0])
        vector_dim = int(header[1])
        print(f"加载词向量文件: {filepath}")
        print(f"词表大小: {vocab_size}, 向量维度: {vector_dim}")

        # 读取每个词的向量
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            # 剩余部分就是向量值
            vector = np.array([float(x) for x in parts[1:]])
            word_vectors[word] = vector

    print(f"成功加载 {len(word_vectors)} 个词的向量")
    return word_vectors


def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def find_similar_words(word_vectors, query_word, top_n=50):
    """查找与目标词最相似的前top_n个词

    参数:
        word_vectors: 词向量字典 {word: numpy_array}
        query_word: 查询词
        top_n: 返回最相似词的个数，默认50
    """
    # 检查查询词是否在词表中
    if query_word not in word_vectors:
        print(f"错误: 词 '{query_word}' 不在词表中!")
        print(f"词表中包含 {len(word_vectors)} 个词（前20个）: {list(word_vectors.keys())[:20]}")
        return

    query_vec = word_vectors[query_word]

    # 计算所有词与查询词的余弦相似度
    similarities = []
    for w, v in word_vectors.items():
        if w != query_word:
            sim = cosine_similarity(query_vec, v)
            similarities.append((w, sim))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 输出结果
    print(f"\n与 '{query_word}' 最相似的前 {top_n} 个词:")
    print("-" * 40)
    print(f"{'排名':<6} {'词汇':<20} {'相似度':<10}")
    print("-" * 40)

    for i, (w, sim) in enumerate(similarities[:top_n], 1):
        print(f"{i:<6} {w:<20} {sim:.6f}")

    print("-" * 40)
    print(f"共找到 {len(similarities)} 个不同的词")


def main():
    """主函数"""
    # 手动输入词向量文件路径
    vector_file = input("请输入词向量文件路径: ").strip()

    if not vector_file:
        print("错误: 文件路径不能为空!")
        sys.exit(1)

    if not os.path.exists(vector_file):
        print(f"错误: 找不到词向量文件 {vector_file}")
        sys.exit(1)

    # 加载词向量
    word_vectors = load_word_vectors(vector_file)

    # 循环查询
    print("\n" + "=" * 50)
    print("词向量相似度查询工具")
    print("输入一个词，即可找到与其最相似的前50个词")
    print("输入 'quit' 或 'exit' 退出程序")
    print("=" * 50)

    while True:
        try:
            query = input("\n请输入查询词: ").strip()

            if query.lower() in ('quit', 'exit', 'q'):
                print("再见!")
                break

            if not query:
                print("请输入有效的词!")
                continue

            find_similar_words(word_vectors, query, top_n=50)

        except KeyboardInterrupt:
            print("\n\n程序被用户中断，再见!")
            break
        except EOFError:
            print("\n\n输入流结束，再见!")
            break
        except Exception as e:
            print(f"发生错误: {e}")


if __name__ == '__main__':
    main()