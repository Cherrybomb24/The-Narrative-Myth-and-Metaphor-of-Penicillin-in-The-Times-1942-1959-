#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""词向量异常词清理 + 分布可视化工具
功能：加载词向量文件 → 自动识别并删除异常词 → 重新计算分布 → 绘图
"""
import numpy as np
import sys
import os
from numpy import linalg as LA

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: 未安装 matplotlib，无法绘图。请运行: pip install matplotlib")


def load_word_vectors(filepath):
    """加载Word2Vec格式的词向量文件"""
    word_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split()
        vocab_size = int(header[0])
        vector_dim = int(header[1])
        print(f"加载词向量文件: {filepath}")
        print(f"词表大小: {vocab_size}, 向量维度: {vector_dim}")

        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            word_vectors[word] = vector

    print(f"成功加载 {len(word_vectors)} 个词的向量")
    return word_vectors


def l2_normalize_vectors(word_vectors):
    """对所有向量做L2归一化"""
    normalized = {}
    for word, vec in word_vectors.items():
        norm = LA.norm(vec)
        if norm == 0:
            normalized[word] = vec
        else:
            normalized[word] = vec / norm
    return normalized


def find_outlier_words(word_vectors, top_k=50):
    """
    识别异常词：计算每个词与所有其他词的余弦相似度，
    统计"极端相似度"（|sim| > 0.95）的比例，比例最高的即为异常词。
    """
    print("\n正在扫描异常词（这可能需要几分钟）...")
    words = list(word_vectors.keys())
    n = len(words)
    vectors = np.array(list(word_vectors.values()))  # (n, dim)

    # 批量计算相似度矩阵（注意内存：n=50000 时矩阵约 10GB，需分批）
    batch_size = 2000
    extreme_ratios = np.zeros(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # 计算 batch 中每个词与所有词的相似度
        batch_vecs = vectors[start:end]          # (batch, dim)
        sims = batch_vecs @ vectors.T             # (batch, n)

        # 排除自身相似度（对角线置0）
        for i in range(end - start):
            sims[i, start + i] = 0.0

        # 极端相似度比例
        extreme_mask = np.abs(sims) > 0.95
        extreme_ratios[start:end] = extreme_mask.mean(axis=1)

        if (start // batch_size + 1) % 5 == 0 or end == n:
            print(f"  进度: {end}/{n}")

    # 按极端比例排序
    indices = np.argsort(extreme_ratios)[::-1]
    outlier_list = [(words[i], extreme_ratios[i]) for i in indices[:top_k]]

    print(f"\n极端相似度比例最高的 {top_k} 个词（建议删除）:")
    print("-" * 55)
    for w, score in outlier_list:
        print(f"  {w:30s}  极端比例: {score:.4f}")
    print("-" * 55)

    return [w for w, _ in outlier_list]


def remove_words(word_vectors, words_to_remove):
    """从词表中删除指定词"""
    removed = 0
    for w in words_to_remove:
        if w in word_vectors:
            del word_vectors[w]
            removed += 1
    print(f"已删除 {removed} 个异常词，剩余 {len(word_vectors)} 个词")
    return word_vectors


def compute_similarity_distribution(word_vectors, sample_size=200):
    """随机采样计算相似度分布"""
    words = list(word_vectors.keys())
    if len(words) < sample_size:
        sample_size = len(words)
        print(f"词表较小，采样全部 {sample_size} 个词")

    sampled_words = np.random.choice(words, sample_size, replace=False)
    all_sims = []

    for i, w in enumerate(sampled_words):
        if (i + 1) % 50 == 0:
            print(f"  计算进度: {i + 1}/{sample_size}")
        vec = word_vectors[w]
        for w2, v2 in word_vectors.items():
            if w2 != w:
                sim = np.dot(vec, v2)
                all_sims.append(sim)

    return np.array(all_sims)


def plot_distribution(sims, title="词向量相似度分布（清理后）"):
    """绘制相似度分布直方图"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，跳过绘图")
        return

    plt.figure(figsize=(12, 6))
    plt.hist(sims, bins=100, density=True, color='steelblue', edgecolor='white', alpha=0.85)
    plt.xlabel('余弦相似度', fontsize=12)
    plt.ylabel('密度', fontsize=12)
    plt.title(title, fontsize=14)
    plt.axvline(x=np.mean(sims), color='red', linestyle='--', label=f'均值={np.mean(sims):.4f}')
    plt.axvline(x=np.median(sims), color='green', linestyle='--', label=f'中位数={np.median(sims):.4f}')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()


def print_stats(sims, label="清理后"):
    """打印分布统计"""
    print(f"\n{'=' * 50}")
    print(f"相似度分布统计 [{label}]")
    print(f"{'=' * 50}")
    print(f"样本对数:     {len(sims):,}")
    print(f"均值:         {np.mean(sims):.6f}")
    print(f"中位数:       {np.median(sims):.6f}")
    print(f"标准差:       {np.std(sims):.6f}")
    print(f"最小值:       {np.min(sims):.6f}")
    print(f"最大值:       {np.max(sims):.6f}")
    print(f"90分位:       {np.percentile(sims, 90):.6f}")
    print(f"95分位:       {np.percentile(sims, 95):.6f}")
    print(f"99分位:       {np.percentile(sims, 99):.6f}")
    print(f"{'=' * 50}")

    if np.percentile(sims, 99) > 0.95:
        print("\n⚠️  警告: 99分位相似度仍超过0.95，可能还需继续清理或重新训练。")
    else:
        print("\n✅ 相似度分布正常。")


def main():
    # 1. 输入文件路径
    vector_file = input("请输入词向量文件路径: ").strip()
    if not vector_file:
        print("错误: 文件路径不能为空!")
        sys.exit(1)
    if not os.path.exists(vector_file):
        print(f"错误: 找不到文件 {vector_file}")
        sys.exit(1)

    # 2. 加载词向量
    word_vectors = load_word_vectors(vector_file)
    if len(word_vectors) == 0:
        print("错误: 词表为空!")
        sys.exit(1)

    # 3. L2归一化
    print("\n正在对向量做L2归一化...")
    word_vectors = l2_normalize_vectors(word_vectors)

    # 4. 【清理前】先算一次分布作为对比
    sample_size = min(200, len(word_vectors))
    print(f"\n[清理前] 正在随机采样 {sample_size} 个词计算相似度分布...")
    sims_before = compute_similarity_distribution(word_vectors, sample_size=sample_size)
    print_stats(sims_before, label="清理前")
    if HAS_MATPLOTLIB:
        plot_distribution(sims_before, title="词向量相似度分布（清理前）")

    # 5. 识别并删除异常词
    outlier_words = find_outlier_words(word_vectors, top_k=50)

    # 让用户确认删除数量
    num_to_remove = input(f"\n确认删除以上 {len(outlier_words)} 个异常词？(y/n): ").strip().lower()
    if num_to_remove != 'y':
        print("已取消删除。")
        sys.exit(0)

    word_vectors = remove_words(word_vectors, outlier_words)

    # 6. 【清理后】重新计算分布
    sample_size_after = min(200, len(word_vectors))
    print(f"\n[清理后] 正在随机采样 {sample_size_after} 个词计算相似度分布...")
    sims_after = compute_similarity_distribution(word_vectors, sample_size=sample_size_after)
    print_stats(sims_after, label="清理后")
    if HAS_MATPLOTLIB:
        plot_distribution(sims_after, title="词向量相似度分布（清理后）")

    # 7. 可选：保存清理后的词向量文件
    save = input("\n是否保存清理后的词向量文件？(y/n): ").strip().lower()
    if save == 'y':
        output_path = vector_file.rsplit('.', 1)[0] + '_cleaned.txt'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"{len(word_vectors)} {next(iter(word_vectors.values())).shape[0]}\n")
            for word, vec in word_vectors.items():
                f.write(word + " " + " ".join(f"{x:.6f}" for x in vec) + "\n")
        print(f"已保存到: {output_path}")


if __name__ == '__main__':
    main()