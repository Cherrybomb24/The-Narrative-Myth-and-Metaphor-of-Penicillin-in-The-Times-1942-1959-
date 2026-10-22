#!/usr/bin/env python3
"""
Robust_Word2Vec_Multifile.py
稳健版的Word2Vec系统 - 修复数学计算错误
"""

import os
import sys
import math
import random
import re
import json
import time
import csv
from typing import List, Dict, Tuple, Any, Optional, Generator
from collections import Counter, defaultdict
from datetime import datetime


# ======================== 修正的向量类 ========================

class RobustVector:
    """修正的向量类，避免数学计算错误"""

    def __init__(self, size: int, init_type: str = 'random'):
        self.size = size
        if init_type == 'random':
            # 均匀随机初始化，避免零向量
            self.data = [random.uniform(-0.1, 0.1) for _ in range(size)]

            # 确保向量有足够的范围
            norm = self.norm()
            if norm < 0.01:
                self.data = [random.uniform(-0.5, 0.5) for _ in range(size)]

        elif init_type == 'zero':
            self.data = [0.0] * size
            # 零向量添加微小噪声避免除零
            for i in range(size):
                self.data[i] += random.uniform(-0.001, 0.001)

        elif init_type == 'one':
            self.data = [1.0] * size

    def _safe_sigmoid(self, x: float) -> float:
        """安全的sigmoid计算"""
        if x > 10:
            return 1.0 - 1e-8  # 避免1.0
        elif x < -10:
            return 1e-8  # 避免0.0
        return 1.0 / (1.0 + math.exp(-x))

    def _safe_log(self, x: float) -> float:
        """安全的log计算"""
        if x <= 0:
            return math.log(1e-10)  # 避免log(0)或负数
        elif x >= 1:
            return math.log(1 - 1e-10)  # 避免log(0)
        return math.log(x)

    def dot(self, other) -> float:
        """点积计算"""
        return sum(a * b for a, b in zip(self.data, other.data))

    def cosine_similarity(self, other) -> float:
        """余弦相似度，避免除零"""
        dot_product = self.dot(other)
        norm_self = self.norm()
        norm_other = other.norm()

        if norm_self < 1e-8 or norm_other < 1e-8:
            return 0.0

        similarity = dot_product / (norm_self * norm_other)

        # 限制范围
        return max(-1.0, min(1.0, similarity))

    def norm(self) -> float:
        """向量模长"""
        return math.sqrt(max(sum(x * x for x in self.data), 1e-8))

    def normalize(self):
        """安全的归一化"""
        n = self.norm()
        if n > 0:
            self.data = [x / n for x in self.data]
        return self

    def add_scaled(self, other, scale: float):
        """添加缩放后的向量"""
        for i in range(self.size):
            self.data[i] += other.data[i] * scale

    def __add__(self, other):
        result = RobustVector(self.size, 'zero')
        for i in range(self.size):
            result.data[i] = self.data[i] + other.data[i]
        return result

    def __sub__(self, other):
        result = RobustVector(self.size, 'zero')
        for i in range(self.size):
            result.data[i] = self.data[i] - other.data[i]
        return result

    def copy(self):
        new_vec = RobustVector(self.size, 'zero')
        new_vec.data = self.data[:]
        return new_vec


# ======================== 文本处理器（简化版） ========================

class TextProcessor:
    """简化的文本处理器"""

    @staticmethod
    def clean_text(text: str) -> str:
        """清理文本"""
        text = text.lower()

        # 保留字母、数字、连字符、撇号
        text = re.sub(r"[^a-z0-9\-\' ]", " ", text)

        # 合并多个空格
        text = " ".join(text.split())

        return text

    @staticmethod
    def process_file(filepath: str) -> List[str]:
        """处理单个文件"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()

            words = TextProcessor.clean_text(text).split()

            # 过滤太短的词
            words = [w for w in words if len(w) > 1]

            print(f"  处理: {os.path.basename(filepath)} -> {len(words)} 个词")
            return words

        except Exception as e:
            print(f"错误处理文件 {filepath}: {e}")
            return []


# ======================== 稳健的Word2Vec ========================

class RobustWord2Vec:
    """修正负样本预测错误的Word2Vec"""

    def __init__(self, vector_size: int = 100, window_size: int = 5,
                 min_count: int = 5, negative: int = 10, learning_rate: float = 0.025):
        self.vector_size = vector_size
        self.window_size = window_size
        self.min_count = min_count
        self.negative = negative
        self.learning_rate = learning_rate

        self.vocab = {}
        self.word_freq = {}
        self.word_vectors = {}
        self.total_words = 0

    def build_vocab(self, sentences: List[List[str]]):
        """构建词汇表"""
        print("构建词汇表...")

        # 统计词频
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(sentence)

        # 过滤低频词
        filtered_words = {word: count for word, count in word_counts.items()
                          if count >= self.min_count}

        # 创建索引
        self.word_freq = dict(sorted(filtered_words.items(), key=lambda x: x[1], reverse=True))
        self.vocab = {word: idx for idx, word in enumerate(self.word_freq.keys())}
        self.total_words = sum(self.word_freq.values())

        print(f"词汇表大小: {len(self.vocab)}")
        print(f"总词数: {self.total_words}")

        # 初始化向量
        self._init_vectors()

        # 构建负采样表
        self._build_negative_table()

        return self

    def _init_vectors(self):
        """初始化词向量"""
        print("初始化词向量...")

        for word in self.vocab:
            self.word_vectors[word] = {
                'in': RobustVector(self.vector_size, 'random'),
                'out': RobustVector(self.vector_size, 'zero')
            }

    def _build_negative_table(self):
        """构建负采样概率表"""
        print("构建负采样表...")

        if not self.word_freq:
            return

        # 计算词的分布概率（按频率^0.75）
        word_list = list(self.word_freq.keys())
        freqs = list(self.word_freq.values())

        # 应用幂律分布
        pow_freqs = [freq ** 0.75 for freq in freqs]
        total_pow = sum(pow_freqs)
        probs = [pf / total_pow for pf in pow_freqs]

        # 构建采样表
        self.neg_table = []
        table_size = 1000000

        cum_prob = 0.0
        prob_multiplier = table_size  # 简化计算

        for idx, prob in enumerate(probs):
            cum_prob += prob
            count = int(prob * prob_multiplier)
            self.neg_table.extend([idx] * count)

        # 确保表格大小
        while len(self.neg_table) < table_size:
            self.neg_table.append(random.randint(0, len(word_list) - 1))

        print(f"负采样表大小: {len(self.neg_table)}")

    def _get_negative_words(self, target_word: str, n: int) -> List[str]:
        """获取负样本词"""
        word_list = list(self.vocab.keys())

        negatives = []
        attempts = 0
        max_attempts = n * 10

        while len(negatives) < n and attempts < max_attempts:
            idx = random.choice(self.neg_table)
            word = word_list[idx]

            if word != target_word and word not in negatives:
                negatives.append(word)

            attempts += 1

        # 如果找不到足够的负样本，补充分数词
        while len(negatives) < n:
            extra_word = random.choice(word_list)
            if extra_word != target_word:
                negatives.append(extra_word)

        return negatives

    def _sigmoid(self, x: float) -> float:
        """安全的sigmoid函数"""
        if x > 10:
            return 0.999999
        elif x < -10:
            return 0.000001
        return 1.0 / (1.0 + math.exp(-x))

    def _log_safe(self, x: float) -> float:
        """安全的log函数"""
        # 限制x的范围在[epsilon, 1-epsilon]
        epsilon = 1e-10
        x = max(epsilon, min(1 - epsilon, x))
        return math.log(x)

    def _train_pair(self, center_word: str, context_word: str, learning_rate: float):
        """训练一对词（中心词和上下文词）"""
        # 获取向量
        center_in = self.word_vectors[center_word]['in']
        context_out = self.word_vectors[context_word]['out']

        # 正样本更新
        dot_pos = center_in.dot(context_out)
        sigmoid_pos = self._sigmoid(dot_pos)

        # 计算梯度（避免NaN）
        error = 1.0 - sigmoid_pos
        if math.isnan(error):
            error = 0.5

        grad = error * learning_rate

        # 更新向量
        center_in.add_scaled(context_out, grad)
        context_out.add_scaled(center_in, grad)

        # 负采样
        negative_words = self._get_negative_words(center_word, self.negative)

        for neg_word in negative_words:
            neg_out = self.word_vectors[neg_word]['out']

            dot_neg = center_in.dot(neg_out)
            sigmoid_neg = self._sigmoid(dot_neg)

            # 负样本的误差
            error_neg = 0.0 - sigmoid_neg
            if math.isnan(error_neg):
                continue

            grad_neg = error_neg * learning_rate

            center_in.add_scaled(neg_out, grad_neg)
            neg_out.add_scaled(center_in, grad_neg)

        return abs(grad)

    def train(self, sentences: List[List[str]], epochs: int = 10):
        """训练模型"""
        print(f"\n开始训练，共{len(sentences)}个句子，{epochs}个epoch")

        total_pairs = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}:")

            epoch_pairs = 0
            epoch_loss = 0.0

            # 调整学习率
            current_lr = self.learning_rate * (1.0 - epoch / epochs)
            current_lr = max(current_lr, 0.0001)

            for i, sentence in enumerate(sentences):
                # 过滤不在词汇表中的词
                valid_words = [w for w in sentence if w in self.vocab]

                if len(valid_words) < 2:
                    continue

                # 生成训练对
                for center_idx, center_word in enumerate(valid_words):
                    # 上下文窗口
                    start = max(0, center_idx - self.window_size)
                    end = min(len(valid_words), center_idx + self.window_size + 1)

                    for context_idx in range(start, end):
                        if context_idx == center_idx:
                            continue

                        context_word = valid_words[context_idx]

                        # 训练一对
                        grad = self._train_pair(center_word, context_word, current_lr)
                        epoch_pairs += 1
                        epoch_loss += grad

                # 显示进度
                if (i + 1) % 100 == 0:
                    print(f"  进度: {i + 1}/{len(sentences)} 句子，{epoch_pairs} 对训练")

            total_pairs += epoch_pairs

            if epoch_pairs > 0:
                avg_loss = epoch_loss / epoch_pairs
                print(f"  本epoch: {epoch_pairs} 训练对，平均梯度: {avg_loss:.6f}")

        print(f"\n训练完成! 总训练对数: {total_pairs}")

        # 对向量进行归一化
        self._normalize_vectors()

    def _normalize_vectors(self):
        """归一化所有输入向量"""
        print("归一化词向量...")

        for word in self.word_vectors:
            self.word_vectors[word]['in'].normalize()

    def get_vector(self, word: str) -> Optional[RobustVector]:
        """获取词的向量"""
        if word not in self.word_vectors:
            return None

        vec = self.word_vectors[word]['in'].copy()
        vec.normalize()  # 返回归一化后的向量
        return vec

    def most_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """查找最相似的词"""
        if word not in self.word_vectors:
            print(f"词 '{word}' 不在词汇表中")
            return []

        target_vec = self.get_vector(word)
        if target_vec is None:
            return []

        similarities = []

        for candidate_word in self.vocab:
            if candidate_word == word:
                continue

            candidate_vec = self.get_vector(candidate_word)
            if candidate_vec:
                similarity = target_vec.cosine_similarity(candidate_vec)
                similarities.append((candidate_word, similarity))

        # 排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:topn]

    def save_vectors(self, filename: str):
        """保存词向量"""
        print(f"保存词向量到 {filename}...")

        with open(filename, 'w', encoding='utf-8') as f:
            # 第一行：词汇量 向量维度
            f.write(f"{len(self.vocab)} {self.vector_size}\n")

            # 每行：词 + 向量
            for word, idx in self.vocab.items():
                vec = self.get_vector(word)
                if vec:
                    vector_str = ' '.join([f"{v:.6f}" for v in vec.data])
                    f.write(f"{word} {vector_str}\n")

        print(f"词向量已保存，共 {len(self.vocab)} 个词")


# ======================== 文件夹处理 ========================

def process_folder(folder_path: str, max_files: int = None, sample_size: int = 1000) -> List[List[str]]:
    """处理文件夹中的所有txt文件"""
    print(f"\n处理文件夹: {folder_path}")

    if not os.path.exists(folder_path):
        print(f"错误: 文件夹不存在 {folder_path}")
        return []

    # 查找所有txt文件
    txt_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                txt_files.append(os.path.join(root, file))

    print(f"找到 {len(txt_files)} 个txt文件")

    if not txt_files:
        print("错误: 没有找到txt文件")
        return []

    # 限制文件数量（如果指定）
    if max_files and len(txt_files) > max_files:
        print(f"只处理前 {max_files} 个文件")
        txt_files = txt_files[:max_files]

    # 处理文件
    all_sentences = []
    file_count = 0

    for filepath in txt_files:
        file_count += 1
        print(f"\n[{file_count}/{len(txt_files)}] 处理: {os.path.basename(filepath)}")

        words = TextProcessor.process_file(filepath)

        if len(words) > 0:
            # 将语料分成句子（简单的句子分割）
            sentences = _split_into_sentences(words, max_length=50)
            all_sentences.extend(sentences)

            print(f"  -> {len(sentences)} 个句子，{len(words)} 个词")

        # 如果积累了太多句子，可以先处理一部分
        if len(all_sentences) > sample_size and sample_size > 0:
            print(f"已达到样本大小 {sample_size}，停止读取")
            break

    print(f"\n处理完成！")
    print(f"总句子数: {len(all_sentences)}")
    print(f"总词数: {sum(len(s) for s in all_sentences)}")

    return all_sentences


def _split_into_sentences(words: List[str], max_length: int = 50) -> List[List[str]]:
    """将词列表分割成句子"""
    sentences = []
    current_sentence = []

    # 简单的句子分割（基于标点）
    sentence_enders = {'.', '!', '?', ';', ':'}

    for word in words:
        current_sentence.append(word)

        # 检查句子结束或达到最大长度
        if (word in sentence_enders or len(current_sentence) >= max_length):
            if len(current_sentence) >= 3:  # 只保留有实质内容的句子
                sentences.append(current_sentence)
            current_sentence = []

    # 添加最后一个句子
    if len(current_sentence) >= 3:
        sentences.append(current_sentence)

    return sentences


# ======================== 词汇导出 ========================

def export_vocabulary_stats(model: RobustWord2Vec, output_file: str):
    """导出词汇统计"""
    print(f"\n导出词汇统计到 {output_file}...")

    vocab_items = list(model.vocab.items())
    vocab_items.sort(key=lambda x: model.word_freq[x[0]], reverse=True)

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['排名', '词', '词频', '频率百分比', '频率对数'])

        total_words = model.total_words

        for idx, (word, _) in enumerate(vocab_items, 1):
            freq = model.word_freq[word]
            percent = (freq / total_words * 100) if total_words > 0 else 0
            log_freq = math.log(freq) if freq > 0 else 0

            writer.writerow([idx, word, freq, f"{percent:.4f}%", f"{log_freq:.2f}"])

    print(f"词汇统计已导出，共 {len(model.vocab)} 个词")


def export_similarity_table(model: RobustWord2Vec, target_word: str,
                            output_file: str, topn: int = 100):
    """导出相似词向量表"""
    print(f"\n导出 '{target_word}' 的相似词向量表...")

    if target_word not in model.vocab:
        print(f"错误: 词 '{target_word}' 不在词汇表中")
        return

    similar_words = model.most_similar(target_word, topn=topn)

    if not similar_words:
        print(f"没有找到与 '{target_word}' 相似的词")
        return

    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['排名', '词', '相似度', '词频', '向量预览'])

        target_vec = model.get_vector(target_word)

        for idx, (word, similarity) in enumerate(similar_words, 1):
            freq = model.word_freq.get(word, 0)
            vec = model.get_vector(word)

            if vec:
                # 向量预览（前5维）
                preview = ', '.join([f"{v:.3f}" for v in vec.data[:5]])

                # 计算与目标词的距离
                if target_vec:
                    dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target_vec.data, vec.data)))
                    writer.writerow([idx, word, f"{similarity:.6f}", freq, preview, f"{dist:.4f}"])
                else:
                    writer.writerow([idx, word, f"{similarity:.6f}", freq, preview])

    print(f"相似词向量表已导出，共 {len(similar_words)} 个词")


# ======================== 主程序 ========================

def main():
    """主程序"""
    print("稳健的Word2Vec词向量系统")
    print("=" * 60)

    # 配置
    VECTOR_SIZE = 100  # 向量维度
    WINDOW_SIZE = 5  # 上下文窗口
    MIN_COUNT = 20  # 最低词频
    NEGATIVE = 10  # 负采样数量
    EPOCHS = 15  # 训练轮数
    LEARNING_RATE = 0.025  # 学习率

    # 1. 选择语料库文件夹
    print("\n1. 选择语料库文件夹")
    folder_path = input("请输入包含txt文件的文件夹路径: ").strip()

    if not os.path.exists(folder_path):
        print("文件夹不存在，请检查路径")
        return

    # 2. 处理文件
    print("\n2. 处理文本文件...")
    sentences = process_folder(folder_path, max_files=None, sample_size=100000)

    if not sentences:
        print("没有可用的句子，请检查文件内容")
        return

    print(f"\n样本句子示例:")
    for i in range(min(3, len(sentences))):
        print(f"  句子 {i + 1}: {' '.join(sentences[i][:10])}...")

    # 3. 训练模型
    print(f"\n3. 训练Word2Vec模型")
    print(f"   向量维度: {VECTOR_SIZE}")
    print(f"   窗口大小: {WINDOW_SIZE}")
    print(f"   最低词频: {MIN_COUNT}")
    print(f"   训练轮数: {EPOCHS}")

    model = RobustWord2Vec(
        vector_size=VECTOR_SIZE,
        window_size=WINDOW_SIZE,
        min_count=MIN_COUNT,
        negative=NEGATIVE,
        learning_rate=LEARNING_RATE
    )

    print("\n开始训练...")
    start_time = time.time()

    model.build_vocab(sentences)
    model.train(sentences, epochs=EPOCHS)

    training_time = time.time() - start_time
    print(f"\n训练完成! 耗时: {training_time / 60:.1f} 分钟")
    print(f"词汇量: {len(model.vocab)}")

    # 4. 导出词汇统计
    print("\n4. 导出词汇统计")
    vocab_file = f"vocabulary_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    export_vocabulary_stats(model, vocab_file)

    # 5. 交互查询
    print("\n" + "=" * 60)
    print("5. 交互式查询模式")
    print("输入 'quit' 退出，'export <词>' 导出相似词表")

    while True:
        query = input("\n>> ").strip()

        if query.lower() == 'quit':
            break

        elif query.lower().startswith('export '):
            target_word = query[7:].strip()
            if target_word:
                output_file = f"{target_word}_similarity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                export_similarity_table(model, target_word, output_file, topn=50)

        elif query:
            # 查询相似词
            similar = model.most_similar(query, topn=15)

            if similar:
                print(f"\n与 '{query}' 最相似的词:")
                for i, (word, sim) in enumerate(similar, 1):
                    freq = model.word_freq.get(word, 0)
                    print(f"  {i:2d}. {word:<20} 相似度: {sim:.4f} 词频: {freq}")
            else:
                print(f"词 '{query}' 不在词汇表中")

    # 6. 保存完整向量
    print("\n6. 保存完整词向量")
    save_option = input("是否保存完整词向量? (y/n): ").lower()

    if save_option == 'y':
        vectors_file = f"word_vectors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        model.save_vectors(vectors_file)
        print(f"词向量已保存到: {vectors_file}")

    print("\n程序完成!")


def quick_test():
    """快速测试函数"""
    print("快速测试模式...")

    # 创建示例数据
    sample_sentences = [
        ["penicillin", "is", "an", "antibiotic", "used", "to", "treat", "bacterial", "infections"],
        ["antibiotics", "like", "penicillin", "kill", "bacteria", "or", "inhibit", "their", "growth"],
        ["bacterial", "infections", "can", "be", "treated", "with", "various", "antibiotics"],
        ["penicillin", "was", "discovered", "by", "alexander", "fleming", "in", "nineteen", "twenty", "eight"],
        ["some", "bacteria", "have", "developed", "resistance", "to", "penicillin"],
        ["the", "discovery", "of", "penicillin", "revolutionized", "modern", "medicine"],
        ["antibiotic", "resistance", "is", "a", "major", "public", "health", "concern"],
        ["penicillin", "is", "derived", "from", "the", "penicillium", "fungus"],
        ["different", "types", "of", "antibiotics", "have", "different", "mechanisms", "of", "action"],
        ["bacteria", "can", "be", "gram", "positive", "or", "gram", "negative"]
    ]

    # 训练模型
    model = RobustWord2Vec(
        vector_size=50,
        window_size=5,  # 3→5
        min_count=30,  # 1→20，关键！
        negative=15,  # 5→15
        learning_rate=0.025
    )

    print("\n词汇表构建中...")
    model.build_vocab(sample_sentences)

    print("\n训练模型...")
    model.train(sample_sentences, epochs=10)

    # 测试相似词
    print("\n测试相似词查询:")

    test_words = ["penicillin", "antibiotic", "bacteria", "infection"]

    for word in test_words:
        print(f"\n与 '{word}' 最相似的词:")
        similar = model.most_similar(word, topn=8)
        for i, (w, sim) in enumerate(similar, 1):
            print(f"  {i}. {w:<20} {sim:.4f}")

    # 导出示例
    export_similarity_table(model, "penicillin", "penicillin_similar_test.csv", topn=10)

    print("\n快速测试完成!")


def demo_folder_processing():
    """演示文件夹处理"""
    print("Word2Vec文件夹处理演示")
    print("=" * 60)

    # 1. 处理文件夹
    folder_path = input("请输入txt文件夹路径: ").strip()

    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    # 2. 参数设置
    print("\n配置参数:")
    vector_size = int(input("向量维度 (默认50): ") or "50")
    window_size = int(input("窗口大小 (默认5): ") or "5")
    min_count = int(input("最低词频 (默认5): ") or "5")
    epochs = int(input("训练轮数 (默认10): ") or "10")

    # 3. 处理文件
    print(f"\n处理文件夹中的文件...")
    sentences = process_folder(folder_path, max_files=50, sample_size=1000)

    if len(sentences) < 10:
        print("错误: 句子太少，无法训练")
        return

    # 4. 训练模型
    print(f"\n训练模型...")

    model = RobustWord2Vec(
        vector_size=vector_size,
        window_size=window_size,
        min_count=min_count,
        negative=10,
        learning_rate=0.025
    )

    model.build_vocab(sentences)
    model.train(sentences, epochs=epochs)

    print(f"\n模型训练完成!")
    print(f"词汇量: {len(model.vocab)}")

    # 5. 演示查询
    print("\n测试查询:")
    while True:
        word = input("\n查询词 (或输入 'quit' 退出): ").strip()

        if word.lower() == 'quit':
            break

        if word in model.vocab:
            similar = model.most_similar(word, topn=10)

            if similar:
                print(f"\n与 '{word}' 最相似的词:")
                for i, (w, sim) in enumerate(similar, 1):
                    freq = model.word_freq.get(w, 0)
                    print(f"  {i}. {w:<20} 相似度: {sim:.4f} 词频: {freq}")

            # 询问是否导出
            export = input(f"\n是否导出 '{word}' 的相似词表? (y/n): ").lower()
            if export == 'y':
                output_file = f"{word}_similar_words.csv"
                export_similarity_table(model, word, output_file, topn=50)
                print(f"已导出到: {output_file}")
        else:
            print(f"词 '{word}' 不在词汇表中")

    print("\n演示完成!")


if __name__ == "__main__":
    print("Word2Vec词向量系统")
    print("=" * 60)
    print("1. 完整处理 (推荐)")
    print("2. 快速测试")
    print("3. 文件夹演示")

    choice = input("\n请选择 (1-3): ").strip()

    if choice == "1":
        main()
    elif choice == "2":
        quick_test()
    elif choice == "3":
        demo_folder_processing()
    else:
        print("无效选择，运行完整处理模式")
        main()
