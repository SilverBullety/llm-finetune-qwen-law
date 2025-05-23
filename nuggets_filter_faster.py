# -*- coding: utf-8 -*-
"""qwen_law_filter_enhanced.py"""
import os
import json
import argparse
from datetime import datetime
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score

DATE_FORMAT = "%Y%m%d_%H%M"  # 精确到分钟
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_BATCH_SIZE = 16  # 根据显存调整，A40可尝试32-64

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="法律问答数据筛选")
    parser.add_argument("--dataset_path", type=str, required=True, help="原始数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--result_dir", type=str, default="./results", help="结果目录")
    parser.add_argument("--threshold", type=float, default=0.5, help="质量提升阈值")
    parser.add_argument("--preprocess", action='store_true', help="执行预处理阶段")
    parser.add_argument("--preprocess_path", type=str, help="指定预处理文件路径")
    return parser.parse_args()

def load_model(model_path):
    """优化后的模型加载"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left',
        pad_token='<|endoftext|>'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用float16加速
        # use_flash_attention_2=True,  # 启用Flash Attention
        trust_remote_code=True
    ).eval()
    
    return tokenizer, model

def generate_batch_answers(model, tokenizer, prompts, batch_size=DEFAULT_BATCH_SIZE):
    """批量生成优化"""
    all_answers = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="生成进度"):
        batch_prompts = prompts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_prompts,
            max_length=512,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # 批量解码
        generated_ids = outputs[:, inputs.input_ids.shape[1]:]
        answers = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        all_answers.extend([a.strip() for a in answers])
    
    return all_answers

def batch_evaluate_quality(references, candidates):
    """批量质量评估"""
    _, _, f1 = score(candidates, references, 
                    lang="zh", 
                    model_type="bert-base-chinese",
                    device=DEVICE,
                    batch_size=DEFAULT_BATCH_SIZE*4)  # 评估批处理
    return f1.cpu().numpy()

def preprocess_data(model, tokenizer, dataset, output_path):
    """重构后的批量预处理"""
    # 准备prompts
    task_sample = dataset[0]
    zero_prompts = [s["input"] for s in dataset]
    one_prompts = [
        f"法律案例参考：{task_sample['input']}\n对应分析：{task_sample['output']}\n\n新案例：{s['input']}\n请分析："
        for s in dataset
    ]
    references = [s["output"] for s in dataset]

    # 批量生成
    print("生成零样本回答...")
    zero_answers = generate_batch_answers(model, tokenizer, zero_prompts)
    print("生成单样本回答...")
    one_answers = generate_batch_answers(model, tokenizer, one_prompts)

    # 批量评估
    print("评估质量...")
    zero_scores = batch_evaluate_quality(references, zero_answers)
    one_scores = batch_evaluate_quality(references, one_answers)

    # 构建结果
    results = []
    for i, s in enumerate(dataset):
        results.append({
            "id": s["id"],
            "input": s["input"],
            "output": s["output"],
            "zero_score": float(zero_scores[i]),
            "one_score": float(one_scores[i]),
            "improvement": float(one_scores[i] - zero_scores[i])
        })
    
    # 保存结果
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"预处理完成，处理样本数：{len(results)}")
    return results

def filter_data(preprocessed, threshold):
    """增强的筛选逻辑"""
    valid_samples = [s for s in preprocessed if "improvement" in s]
    errors = [s for s in preprocessed if "error" in s]
    
    print("\n" + "="*40)
    print("筛选统计：")
    print(f"总样本数：{len(preprocessed)}")
    print(f"有效样本：{len(valid_samples)}")
    print(f"错误样本：{len(errors)}")
    
    passed = [s for s in valid_samples if s["improvement"] >= threshold]
    
    if passed:
        print("\n质量提升分布：")
        print(f"最大值：{max(s['improvement'] for s in passed):.4f}")
        print(f"最小值：{min(s['improvement'] for s in passed):.4f}")
        print(f"平均值：{sum(s['improvement'] for s in passed)/len(passed):.4f}")
        print(f"通过样本数：{len(passed)}")
    else:
        print("\n警告：没有样本通过当前阈值")
    
    return [s for s in passed if "input" in s]

def get_default_preprocess_path(result_dir):
    """生成默认预处理路径"""
    timestamp = datetime.now().strftime(DATE_FORMAT)
    return os.path.join(result_dir, f"preprocessed_{timestamp}.json")

def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    
    if args.preprocess:
        # 预处理模式
        preprocess_path = args.preprocess_path or get_default_preprocess_path(args.result_dir)
        print(f"开始预处理，结果将保存至：{preprocess_path}")
        
        tokenizer, model = load_model(args.model_path)
        
        with open(args.dataset_path, "r") as f:
            dataset = [json.loads(line) for line in f]
        
        dataset = dataset[-5000:]

        preprocess_data(model, tokenizer, dataset, preprocess_path)
        args.preprocess_path = preprocess_path
        return

    # 筛选模式
    preprocess_path = args.preprocess_path or get_default_preprocess_path(args.result_dir)
    
    if not os.path.exists(preprocess_path):
        available_files = "\n".join(f for f in os.listdir(args.result_dir) if f.startswith("preprocessed"))
        raise FileNotFoundError(
            f"未找到预处理文件，请先运行预处理阶段\n可用文件：\n{available_files or '无'}"
        )
    
    print(f"正在加载预处理文件：{preprocess_path}")
    with open(preprocess_path, "r") as f:
        preprocessed = json.load(f)
    
    passed_samples = filter_data(preprocessed, args.threshold)
    
    # 保存结果
    timestamp = datetime.now().strftime(DATE_FORMAT)
    result_path = os.path.join(args.result_dir, f"result_{timestamp}.json")
    with open(result_path, "w") as f:
        json.dump(passed_samples, f, ensure_ascii=False, indent=2)
    
    print(f"\n筛选结果已保存至：{result_path}")

if __name__ == "__main__":
    main()