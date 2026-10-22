import argparse
import os
import re


def is_float(s):
    """判断字符串是否为浮点数"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def fix_and_convert(input_file, output_vectors, output_meta):
    print(f" 正在读取并修复文件: {input_file}")

    words = []
    vectors = []

    current_word = None
    current_vector = []
    target_dim = 100  # 预设维度

    # 用于统计
    total_lines = 0
    fixed_breaks = 0
    skipped = 0

    # 匹配标准科学计数法或浮点数
    float_pattern = re.compile(r'^[-+]?\d*\.\d+([eE][-+]?\d+)?$')

    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue

            tokens = line.split()

            # 判断第一个 token 是词还是被截断的数字
            first_token_is_word = not float_pattern.match(tokens[0]) and not (is_float(tokens[0]) and '.' in tokens[0])

            # 特殊处理：如果第一个词就是纯数字（如 "12"），且当前没有正在处理的词，或者当前向量已满
            if is_float(tokens[0]) and '.' not in tokens[0] and (
                    current_word is None or len(current_vector) >= target_dim):
                first_token_is_word = True

            if first_token_is_word:
                # 遇到新词，保存上一个词的数据
                if current_word is not None:
                    if len(current_vector) == target_dim:
                        words.append(current_word)
                        vectors.append(current_vector)
                    else:
                        skipped += 1

                # 开始新词
                current_word = tokens[0]
                current_vector = []
                # 将本行剩余部分加入向量
                for t in tokens[1:]:
                    if is_float(t):
                        current_vector.append(float(t))
            else:
                # 遇到续行（被截断的向量），拼接到当前向量中
                fixed_breaks += 1
                for t in tokens:
                    if is_float(t):
                        current_vector.append(float(t))

        # 处理文件末尾的最后一个词
        if current_word is not None and len(current_vector) == target_dim:
            words.append(current_word)
            vectors.append(current_vector)
        elif current_word is not None:
            skipped += 1

    print(f" 修复完成！")
    print(f"   - 扫描总行数: {total_lines}")
    print(f"   - 修复断行次数: {fixed_breaks}")
    print(f"   - 成功提取词向量: {len(words)} 个")
    print(f"   - 丢弃异常行: {skipped} 个")

    if not words:
        print(" 错误：没有提取到任何有效的词向量，请检查文件内容。")
        return

    # 1. 保存向量文件 (Tensors) - 纯数字，Tab分隔
    print(f" 正在保存向量文件: {output_vectors}")
    with open(output_vectors, 'w', encoding='utf-8') as f:
        for vec in vectors:
            f.write('\t'.join(map(str, vec)) + '\n')

    # 2. 保存元数据文件 (Metadata) - 纯词汇，每行一个
    print(f" 正在保存标签文件: {output_meta}")
    with open(output_meta, 'w', encoding='utf-8') as f:
        for w in words:
            f.write(w + '\n')

    print(" 全部完成！请将生成的向量文件和标签文件一起上传到 TensorFlow Projector。")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="修复截断的词向量并转换为TSV")
    # 所有参数都设为可选，不传就进入交互模式
    parser.add_argument('--file', default=None, help='原始 txt 词向量文件路径')
    parser.add_argument('--output_vectors', default=None, help='输出的向量 TSV 文件名')
    parser.add_argument('--output_meta', default=None, help='输出的标签 TSV 文件名')

    args = parser.parse_args()

    # 交互模式：询问输入文件路径
    if args.file is None:
        input_path = input("\n 请输入原始 txt 文件的路径（可直接拖拽文件到此处）：").strip()
        # 去除拖拽时可能产生的引号
        input_path = input_path.strip('"').strip("'")
        if not input_path:
            print(" 未输入路径，已退出。")
            exit()
        args.file = input_path

    # 自动推导输出路径：默认在输入文件同目录下，以原文件名+output命名
    input_dir = os.path.dirname(os.path.abspath(args.file))
    # 获取不带后缀的原文件名，例如 "261020整体样本词向量数据"
    base_name = os.path.splitext(os.path.basename(args.file))[0]

    if args.output_vectors is None:
        args.output_vectors = os.path.join(input_dir, f'{base_name}_output_vectors.tsv')
    if args.output_meta is None:
        args.output_meta = os.path.join(input_dir, f'{base_name}_output_meta.tsv')

    if not os.path.exists(args.file):
        print(f" 找不到文件: {args.file}")
    else:
        fix_and_convert(args.file, args.output_vectors, args.output_meta)
