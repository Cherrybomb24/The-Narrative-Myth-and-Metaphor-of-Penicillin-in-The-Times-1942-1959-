import os
import re
from tkinter import Tk, filedialog, messagebox

# 所有要过滤的语法词（完整列表）
GRAMMAR_WORDS = {
    # 所有格代词
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'this', 'that', 'these', 'those',
    'which', 'what', 'who', 'whom', 'whose',

    # 介词
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'about', 'between', 'under', 'over', 'through',
    'after', 'before', 'during', 'since', 'until', 'without',
    'above', 'below', 'around', 'among', 'across',

    # 冠词
    'the', 'a', 'an',

    # 连词
    'and', 'or', 'but', 'so', 'because', 'if',
    'when', 'although', 'though', 'while', 'since',
    'unless', 'until', 'nor', 'yet',

    # 感叹词
    'oh', 'ah', 'wow', 'oops', 'ouch', 'hey', 'hi',

    # 常见助动词
    'am', 'is', 'are', 'was', 'were', 'be', 'being', 'been',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing',
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    # 自己加的
    'as','not','mr','there','all','one','no','more','other','than','some','any',''
}




def remove_grammar_words(text):
    """真正去掉所有语法词"""
    # 分步骤处理
    lines = text.split('\n')
    result_lines = []

    for line in lines:
        if not line.strip():
            result_lines.append('')
            continue

        # 创建一个正则表达式模式，匹配单独出现的语法词（忽略大小写）
        # 模式说明：\b 表示单词边界
        pattern = r'\b(' + '|'.join(GRAMMAR_WORDS) + r')\b'

        # 先处理独立的语法词
        line_clean = re.sub(pattern, '', line, flags=re.IGNORECASE)

        # 再处理缩写形式（如 I'm, don't, doesn't 等）
        contractions = {
            r'\bI\'m\b': '',
            r'\byou\'re\b': '',
            r'\bhe\'s\b': '',
            r'\bshe\'s\b': '',
            r'\bit\'s\b': '',
            r'\bwe\'re\b': '',
            r'\bthey\'re\b': '',
            r'\bdon\'t\b': '',
            r'\bdoesn\'t\b': '',
            r'\bdidn\'t\b': '',
            r'\bisn\'t\b': '',
            r'\baren\'t\b': '',
            r'\bwasn\'t\b': '',
            r'\bweren\'t\b': '',
            r'\bcan\'t\b': '',
            r'\bcouldn\'t\b': '',
            r'\bwon\'t\b': '',
            r'\bwouldn\'t\b': '',
            r'\bshouldn\'t\b': '',
            r'\bI\'ve\b': '',
            r'\byou\'ve\b': '',
            r'\bwe\'ve\b': '',
            r'\bthey\'ve\b': '',
        }

        for pattern, replacement in contractions.items():
            line_clean = re.sub(pattern, replacement, line_clean, flags=re.IGNORECASE)

        # 清理多余的标点和空格
        # 去掉单词间的多余空格
        line_clean = re.sub(r'\s+', ' ', line_clean)
        # 去掉句子开头和结尾的空格
        line_clean = line_clean.strip()
        # 处理标点周围的空格问题
        line_clean = re.sub(r'\s+([.,;:!?])', r'\1', line_clean)
        line_clean = re.sub(r'([.,;:!?])\s+', r'\1 ', line_clean)

        if line_clean:
            result_lines.append(line_clean)

    return '\n'.join(result_lines)


def process_folder(input_folder):
    """处理文件夹中的所有txt文件"""
    # 创建输出文件夹
    output_folder = f"{input_folder}_去语法词"

    # 如果文件夹已存在，添加数字后缀
    counter = 1
    original_name = output_folder
    while os.path.exists(output_folder):
        output_folder = f"{original_name}_{counter}"
        counter += 1

    os.makedirs(output_folder)

    # 查找所有txt文件
    txt_files = []
    for file in os.listdir(input_folder):
        if file.lower().endswith('.txt'):
            txt_files.append(file)

    if not txt_files:
        print(f"没有找到txt文件！")
        return False

    print(f"找到 {len(txt_files)} 个txt文件")

    # 处理每个文件
    processed_count = 0
    for filename in txt_files:
        input_path = os.path.join(input_folder, filename)

        try:
            # 读取文件（尝试不同编码）
            text = None
            for encoding in ['utf-8', 'gbk', 'latin-1', 'utf-16', 'cp1252']:
                try:
                    with open(input_path, 'r', encoding=encoding) as f:
                        text = f.read()
                        # 如果读取成功但没有内容，继续尝试其他编码
                        if text.strip():
                            print(f"✓ 使用 {encoding} 编码成功读取: {filename}")
                            break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    continue

            if not text or not text.strip():
                print(f"✗ 文件为空或无法读取: {filename}")
                continue

            print(f"处理前示例: {text[:100]}...")

            # 去掉语法词
            clean_text = remove_grammar_words(text)

            print(f"处理后示例: {clean_text[:100]}...")
            print("-" * 50)

            # 保存到新文件
            output_path = os.path.join(output_folder, filename)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(clean_text)

            processed_count += 1

        except Exception as e:
            print(f"✗ 处理失败 {filename}: {e}")

    return output_folder, processed_count, len(txt_files)


def main():
    """主函数"""
    # 创建窗口（不显示）
    root = Tk()
    root.withdraw()  # 隐藏主窗口

    print("=" * 50)
    print("语法词过滤工具")
    print("=" * 50)
    print("此工具将删除所有代词、介词、连词、冠词、感叹词、数词")
    print("=" * 50)

    print("\n请选择包含txt文件的文件夹...")

    # 选择文件夹
    folder_path = filedialog.askdirectory(title="选择包含txt文件的文件夹")

    if not folder_path:
        print("用户取消了选择")
        return

    print(f"已选择文件夹: {folder_path}")

    try:
        # 处理文件夹
        result = process_folder(folder_path)
        if not result:
            return

        output_folder, processed, total = result

        if processed > 0:
            print(f"\n✅ 处理完成！")
            print(f"   总文件: {total} 个")
            print(f"   成功: {processed} 个")
            print(f"   输出文件夹: {output_folder}")

            # 显示示例对比
            print(f"\n示例对比：")
            for filename in os.listdir(output_folder)[:2]:  # 显示前2个文件的对比
                if filename.endswith('.txt'):
                    original_path = os.path.join(folder_path, filename.replace('_', ''))
                    new_path = os.path.join(output_folder, filename)

                    try:
                        with open(original_path, 'r', encoding='utf-8') as f:
                            original = f.read()
                        with open(new_path, 'r', encoding='utf-8') as f:
                            new = f.read()

                        if len(original.split()) > 0:
                            removal_rate = (len(original.split()) - len(new.split())) / len(original.split()) * 100
                            print(f"{filename}: 移除了约 {removal_rate:.1f}% 的语法词")
                    except:
                        pass

            # 显示成功消息
            messagebox.showinfo(
                "处理完成",
                f"✓ 语法词过滤完成！\n\n"
                f"输出文件夹:\n{output_folder}\n\n"
                f"总文件: {total} 个\n"
                f"成功处理: {processed} 个\n\n"
                f"现在可以去查看结果了。"
            )
        else:
            print(f"\n❌ 处理失败：没有找到可处理的txt文件")
            messagebox.showerror("处理失败", "没有找到可处理的txt文件")

    except Exception as e:
        error_msg = f"处理过程中出错: {str(e)}"
        print(f"\n❌ {error_msg}")
        messagebox.showerror("处理出错", error_msg)


if __name__ == "__main__":
    main()
