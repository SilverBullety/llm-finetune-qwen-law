"""
qwen7b_sft_training.py
功能：使用监督式微调训练Qwen-7B模型
"""
import json
import os
import torch
from tqdm import tqdm
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
import argparse

# 配置区
DATE_FORMAT = "%Y%m%d_%H%M"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_MAX_LENGTH = 1024  # 根据显存调整（网页[2]显存优化建议）

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen-7B Supervised Fine-tuning")
    parser.add_argument("--model_path", type=str, required=True, help="预训练模型路径")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据集路径（JSONL格式）")
    parser.add_argument("--output_dir", type=str, default="./sft_models", help="模型输出目录")
    parser.add_argument("--logging_dir", type=str, default="./logs", help="日志目录")
    parser.add_argument("--max_length", type=int, default=DEFAULT_MAX_LENGTH, help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮次")
    parser.add_argument("--batch_size", type=int, default=2, help="实际batch_size = batch_size * grad_accum")
    parser.add_argument("--grad_accum", type=int, default=4, help="梯度累积步数")
    parser.add_argument("--lr", type=float, default=2e-5, help="学习率")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="预热比例")
    return parser.parse_args()

def load_and_process_data(tokenizer, file_path, max_length):
    """
    数据处理流程（网页[9][10]数据加载优化）
    返回预处理后的Dataset对象
    """
    samples = []
    with open(file_path, "r") as f:
        for line in tqdm(f, desc="Loading dataset"):
            data = json.loads(line)
            samples.append({
                "input": data["input"],
                "output": data["output"]
            })
    
    def tokenize_function(examples):
        # 正确的数据访问方式（网页[9]批处理规范）
        inputs = examples["input"]
        outputs = examples["output"]
        
        # 构建训练样本格式（网页[3]模板修正）
        prompts = [
            f"Input: {input_text}\nOutput: {output_text}{tokenizer.eos_token}"
            for input_text, output_text in zip(inputs, outputs)
        ]
        
        # 分词处理（保持原有逻辑）
        tokenized = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        # 创建labels（修正索引方式）
        labels = tokenized["input_ids"].clone()
        input_lens = [
            len(tokenizer(f"Input: {input_text}\nOutput: ").input_ids)
            for input_text in inputs  # 直接使用输入列表
        ]
        
        for i, input_len in enumerate(input_lens):
            labels[i, :input_len] = -100
            labels[i, tokenized.attention_mask[i] == 0] = -100
        
        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels
        }
    
    return Dataset.from_list(samples).map(
        tokenize_function,
        batched=True,
        batch_size=1000,
        remove_columns=["input", "output"]
    )

def main():
    args = parse_args()
    
    # 创建输出目录（网页[8]实验管理规范）
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.logging_dir, exist_ok=True)
    
    # 初始化模型（网页[1]官方加载方式）
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        pad_token="<|endoftext|>"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # 数据准备
    train_dataset = load_and_process_data(tokenizer, args.train_data, args.max_length)
    
    # 训练参数配置（网页[2]混合精度优化）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        fp16=False,
        logging_dir=args.logging_dir,
        save_strategy="epoch",
        evaluation_strategy="no",
        report_to="none",
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        remove_unused_columns=False
    )
    
    # 初始化Trainer（网页[6]大模型训练技巧）
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # 开始训练
    print(f"开始训练，总样本数：{len(train_dataset)}")
    trainer.train()
    
    # 保存最终模型（网页[7]模型保存规范）
    final_dir = os.path.join(args.output_dir, datetime.now().strftime(DATE_FORMAT))
    trainer.save_model(final_dir)
    print(f"训练完成，模型已保存至：{final_dir}")

if __name__ == "__main__":
    main()