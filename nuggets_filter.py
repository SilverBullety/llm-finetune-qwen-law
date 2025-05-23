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

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="法律问答数据筛选")
    parser.add_argument("--dataset_path", type=str, required=True, help="原始数据集路径")
    parser.add_argument("--model_path", type=str, required=True, help="模型路径")
    parser.add_argument("--result_dir", type=str, default="./results", help="结果目录")
    parser.add_argument("--threshold", type=float, default=0.2, help="质量提升阈值")
    parser.add_argument("--preprocess", action='store_true', help="执行预处理阶段")
    parser.add_argument("--preprocess_path", type=str, help="指定预处理文件路径")
    return parser.parse_args()

def load_model(model_path):
    """加载模型和分词器"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='left',
        pad_token='<|endoftext|>'
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    return tokenizer, model

def generate_answer(model, tokenizer, prompt):
    """生成法律答案（改进版）"""
    inputs = tokenizer(
        prompt,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

def evaluate_quality(reference, candidate):
    """质量评估"""
    _, _, f1 = score([candidate], [reference], lang="zh", model_type="bert-base-chinese")
    return f1.mean().item()

def preprocess_data(model, tokenizer, dataset, output_path):
    """增强的预处理流程"""
    task_sample = dataset[0]
    reference = task_sample["output"]
    
    results = []
    progress_bar = tqdm(dataset, desc="预处理进度")
    
    try:
        # 实时写入准备
        with open(output_path, "w", encoding='utf-8') as f:
            f.write("[\n")  # 开始JSON数组
            
            for idx, sample in enumerate(progress_bar):
                try:
                    # 零样本生成
                    zero_answer = generate_answer(model, tokenizer, sample["input"])
                    
                    # 单样本生成
                    prompt = f"法律案例参考：{task_sample['input']}\n对应分析：{task_sample['output']}\n\n新案例：{sample['input']}\n请分析："
                    one_answer = generate_answer(model, tokenizer, prompt)
                    
                    # 质量评估
                    zero_score = evaluate_quality(reference, zero_answer)
                    one_score = evaluate_quality(reference, one_answer)
                    
                    result = {
                        "id": sample["id"],
                        "input": sample["input"],
                        "output": sample["output"],
                        "zero_score": zero_score,
                        "one_score": one_score,
                        "improvement": one_score - zero_score
                    }
                    
                    # 写入当前结果
                    if idx > 0:
                        f.write(",\n")
                    json.dump(result, f, ensure_ascii=False)
                    f.flush()  # 确保实时写入
                    
                    # 进度更新
                    progress_bar.set_postfix({
                        "已处理": idx+1,
                        "当前提升": result["improvement"]
                    })
                    
                    results.append(result)
                    
                except Exception as e:
                    error_msg = f"样本 {sample.get('id', '未知')} 处理失败: {str(e)}"
                    print(f"\n{error_msg}")
                    results.append({"id": sample["id"], "error": error_msg})
            
            f.write("\n]")  # 结束JSON数组
        
        print(f"\n预处理完成，共处理 {len(results)} 个样本")
        return results
    
    except KeyboardInterrupt:
        print("\n用户中断，正在保存已处理数据...")
        with open(output_path, "a", encoding='utf-8') as f:
            f.write("\n]")  # 尝试正常结束JSON数组
        raise

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
        
        preprocess_data(model, tokenizer, dataset, preprocess_path)
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