"""
qwen7b_bertscore_evaluation.py
功能：实现Qwen-7B模型的批量推理和BERTScore评估
"""
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from bert_score import score
import os
from datetime import datetime
import argparse

# 配置区
# MODEL_PATH = "/mnt/sdb/ww/ckpt/qwen-7b"  # 网页[1]模型加载方式
# DATASET_PATH = "/mnt/sdb/ww/datasets/llm/disc-law-sft-subsets/test.jsonl"
# RESULT_DIR = "./inference_results"  # 结果存储目录（用户可自定义）
DATE_FORMAT = "%Y%m%d_%H%M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATION_CONFIG = {  # 网页[1]生成参数调整
    "max_new_tokens": 512,
    "do_sample": True,
    "temperature": 0.7,
    "top_p": 0.95
}

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-7B BERTScore Evaluation")
    parser.add_argument(
        "--dataset_path", 
        type=str, 
        default="/mnt/sdb/ww/datasets/llm/disc-law-sft-subsets/test.jsonl", 
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/mnt/sdb/ww/ckpt/qwen-7b", 
        help="Path to the dataset for evaluation"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/mnt/sdb/ww/ckpt/qwen-7b",
        help="Path to the tokenizer for the model"
    )
    parser.add_argument(
        "--result_dir", 
        type=str, 
        default="./inference_results", 
        help="Directory to save the inference results"
    )
    parser.add_argument("--result_path", type=str, default=None, help="Path to the inference results file, if already generated")
    args = parser.parse_args()
    return args

def get_dynamic_path(args):
    """生成带日期的结果文件路径"""
    current_date = datetime.now().strftime(DATE_FORMAT)
    os.makedirs(args.result_dir, exist_ok=True)  # 自动创建目录
    return os.path.join(
        args.result_dir, 
        f"{current_date}_inference_results.jsonl"
    )


def model_inference(args):
    # 初始化模型（网页[1]加载方式增强）
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
        pad_token="<|endoftext|>"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).eval()
    
    result_path = get_dynamic_path(args)

    # 读取数据集（网页[9][10] JSONL处理方案）
    with open(args.dataset_path, "r") as f, open(result_path, "w") as out_f:
        for line in tqdm(f, desc="Processing samples"):
            data = json.loads(line)
            
            # 生成回复（网页[1]生成逻辑优化）
            inputs = tokenizer(
                data["input"], 
                return_tensors="pt",
                padding=True
            ).to(DEVICE)
            
            outputs = model.generate(
                **inputs,
                **GENERATION_CONFIG,
                eos_token_id=tokenizer.eos_token_id
            )
            
            response = tokenizer.decode(
                outputs[0][len(inputs["input_ids"][0]):], 
                skip_special_tokens=True
            )
            
            # 保存结果（网页[7]数据格式参考）
            out_data = {
                "id": data["id"],
                "input": data["input"],
                "ground_truth": data["output"],
                "generated": response.strip()
            }
            out_f.write(json.dumps(out_data, ensure_ascii=False) + "\n")
    
    return result_path

def bertscore_evaluation(result_path):
    # 加载推理结果
    refs, hyps = [], []
    with open(result_path, "r") as f:
        for line in f:
            data = json.loads(line)
            refs.append(data["ground_truth"])
            hyps.append(data["generated"])
    
    # 计算BERTScore（网页[7]评估实现）
    P, R, F1 = score(
        hyps, 
        refs,
        lang="zh",
        model_type="bert-base-chinese",
        verbose=True,
        device=DEVICE
    )
    
    # 结果分析（网页[6]指标扩展）
    print(f"\n评估结果（{len(refs)}条样本）:")
    print(f"Precision (P): {P.mean().item():.4f}")
    print(f"Recall (R): {R.mean().item():.4f}")
    print(f"F1 Score: {F1.mean().item():.4f}")
    
    # 保存详细报告
    with open("bertscore_report.txt", "w") as f:
        f.write("样本ID | P | R | F1\n")
        with open(result_path, "r") as res_f:
            for idx, line in enumerate(res_f):
                data = json.loads(line)
                f.write(f"{data['id']} | {P[idx]:.4f} | {R[idx]:.4f} | {F1[idx]:.4f}\n")

def main():
    args = parse_args()
    if args.result_path is None:
        args.result_path = model_inference(args)
    bertscore_evaluation(args.result_path)

if __name__ == "__main__":
    main()
    