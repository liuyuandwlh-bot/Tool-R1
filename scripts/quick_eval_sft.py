#!/usr/bin/env python
"""简单的模型评测工具 - 在硬数据集上快速测试 SFT 模型"""

import json
import sys
import random
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def load_samples(data_path, split="train", limit=100):
    """加载样本"""
    samples_file = Path(data_path) / split / "samples.jsonl"
    if not samples_file.exists():
        raise FileNotFoundError(f"数据不存在: {samples_file}")
    
    samples = []
    with open(samples_file) as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            samples.append(json.loads(line))
    
    print(f"✓ 加载了 {len(samples)} 个样本")
    
    # 统计任务分布
    from collections import Counter
    task_counts = Counter(s['task_type'] for s in samples)
    print(f"  任务分布: {dict(task_counts)}")
    
    return samples

def evaluate_sft(model_path, data_path="data/train-hard-v2", limit=50):
    """快速评测 SFT 模型"""
    
    print(f"📦 加载模型: {model_path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
    model.eval()
    
    print(f"✓ 模型已加载到 {device}")
    print()
    
    # 加载硬数据集的 train 分割
    samples = load_samples(data_path, split="train", limit=limit)
    
    print(f"\n🧪 评测 {len(samples)} 个样本...\n")
    
    correct = 0
    results_by_task = {}
    
    for idx, sample in enumerate(samples):
        if (idx + 1) % 10 == 0:
            print(f"  进度: {idx + 1}/{len(samples)}")
        
        task_type = sample.get('task_type', 'unknown')
        if task_type not in results_by_task:
            results_by_task[task_type] = {'correct': 0, 'total': 0}
        
        # 提示词
        prompt = sample['question']
        
        # 生成答案
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0,
                do_sample=False,
                top_p=1.0,
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # 检查答案
        expected_answer = sample.get('answer', '')
        is_correct = expected_answer in response or response in str(expected_answer)
        
        if is_correct:
            correct += 1
        
        results_by_task[task_type]['total'] += 1
        if is_correct:
            results_by_task[task_type]['correct'] += 1
    
    accuracy = correct / len(samples) if samples else 0
    
    print(f"\n{'='*50}")
    print(f"📊 整体结果:")
    print(f"   正确率: {correct}/{len(samples)} = {accuracy:.1%}")
    print()
    print(f"按任务类型分解:")
    for task, stats in sorted(results_by_task.items()):
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"   {task:20s} {stats['correct']:3d}/{stats['total']:3d} = {acc:6.1%}")
    print(f"{'='*50}")
    
    return {
        'overall_accuracy': accuracy,
        'correct': correct,
        'total': len(samples),
        'by_task': {k: v['correct']/v['total'] if v['total']>0 else 0 for k, v in results_by_task.items()}
    }

if __name__ == "__main__":
    model = "outputs/sft-short/merged"
    data = "data/train-hard-v2"
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    
    results = evaluate_sft(model, data, limit=limit)
    
    # 保存结果
    output_file = Path("outputs/quick-eval-results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 结果已保存: {output_file}")
