import json
import os
import random

def split_jsonl_dataset(input_file, output_dir, train_size=50000, test_size=5000, seed=42):
    """
    分割 JSONL 数据集为训练集和测试集
    :param input_file: 输入文件路径
    :param output_dir: 输出目录路径
    :param train_size: 训练集行数 (默认50k)
    :param test_size: 测试集行数 (默认5k)
    :param seed: 随机种子 (默认42)
    """
    # 创建输出目录
    os.makedirs(os.path.join(output_dir, "disc-law-sft-subsets"), exist_ok=True)
    
    # 读取所有数据行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    # 数据随机打乱（保持可复现性）
    random.seed(seed)
    random.shuffle(lines)

    # 分割数据集
    train_data = lines[:train_size]
    test_data = lines[train_size:train_size+test_size]

    # 写入训练集
    with open(os.path.join(output_dir, "disc-law-sft-subsets", "train.jsonl"), 'w') as f:
        for line in train_data:
            f.write(line + '\n')

    # 写入测试集
    with open(os.path.join(output_dir, "disc-law-sft-subsets", "test.jsonl"), 'w') as f:
        for line in test_data:
            f.write(line + '\n')

if __name__ == "__main__":
    dataset_path = '/mnt/sdb/ww/datasets/llm/disc-law-sft/DISC-Law-SFT-Pair-QA-released.jsonl'
    # 使用示例
    split_jsonl_dataset(
        input_file=dataset_path,  # 替换为输入文件路径
        output_dir="/mnt/sdb/ww/datasets/llm",
        train_size=50000,
        test_size=5000
    )