#!/usr/bin/python3
"""
task3 NULL 弃权机制 - 负例挖掘脚本

生成四类负例：
1. 部署匹配负例：用 task2 模型在训练集上的预测（非金标 span）
2. ±1 边界漂移：在金标边界基础上±1 字符
3. 跨目标负例：不同目标词的跨度组合
4. 随机跨度：随机生成的无效跨度

输出格式：与 dataset/cfn-dev.json 相同的 JSON 格式，额外添加 negative_spans 字段
"""

import json
import random
import argparse
from typing import List, Dict, Tuple
import os

def load_json(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, file_path: str):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def generate_boundary_drift(spans: List[Tuple[int, int]], drift_range: int = 1) -> List[Tuple[int, int]]:
    """生成±1 边界漂移负例"""
    drift_spans = []
    for start, end in spans:
        for d in range(-drift_range, drift_range + 1):
            if d == 0:
                continue
            new_start = max(0, start + d)
            new_end = max(0, end + d)
            if new_start != start or new_end != end:
                drift_spans.append((new_start, new_end))
    return drift_spans

def generate_cross_target_negatives(samples: List[Dict], target_pairs: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """生成交叉目标负例：用 A 句的目标词 + B 句的跨度"""
    cross_spans = []
    if len(samples) < 2:
        return cross_spans
    
    for i, sample in enumerate(samples):
        text = sample['text']
        # 随机选择另一个句子的目标词区间
        other_idx = (i + 1) % len(samples)
        other_target_start, other_target_end = samples[other_idx]['target'][0], samples[other_idx]['target'][1]
        
        # 用当前句的跨度，但搭配其他句的目标词
        for start, end in target_pairs:
            if start < len(text) and end <= len(text):
                cross_spans.append((start, end))
    
    return cross_spans

def generate_random_spans(text: str, num_spans: int = 10) -> List[Tuple[int, int]]:
    """生成随机跨度负例"""
    random_spans = []
    text_len = len(text)
    
    for _ in range(num_spans):
        # 随机跨度长度（3-15 字符）
        span_len = random.randint(3, min(15, text_len))
        start = random.randint(0, text_len - span_len)
        end = start + span_len
        random_spans.append((start, end))
    
    return random_spans

def load_task2_predictions(task2_result_file: str) -> Dict[str, List[Tuple[int, int]]]:
    """加载 task2 预测结果"""
    with open(task2_result_file, 'r', encoding='utf-8') as f:
        predictions = json.load(f)
    
    # 按 sentence_id 分组
    spans_by_sentence = {}
    for pred in predictions:
        sent_id = pred[0]
        start, end = pred[1], pred[2]
        if sent_id not in spans_by_sentence:
            spans_by_sentence[sent_id] = []
        spans_by_sentence[sent_id].append((start, end))
    
    return spans_by_sentence

def mine_negative_examples(
    train_file: str,
    task2_result_file: str,
    output_file: str,
    negative_budget_ratio: float = 2.0,
    max_negatives_per_sentence: int = 16
):
    """
    挖掘负例并添加到训练数据中
    
    Args:
        train_file: 原始训练集路径
        task2_result_file: task2 预测结果路径
        output_file: 输出文件路径
        negative_budget_ratio: 负例预算与正例的比例（默认 2 倍）
        max_negatives_per_sentence: 每句最大负例数（默认 16）
    """
    print(f"加载训练数据：{train_file}")
    train_data = load_json(train_file)
    
    print(f"加载 task2 预测结果：{task2_result_file}")
    task2_spans = load_task2_predictions(task2_result_file)
    
    print("开始挖掘负例...")
    total_positive = 0
    total_negative = 0
    negative_stats = {
        'deployment_match': 0,
        'boundary_drift': 0,
        'cross_target': 0,
        'random_span': 0
    }
    
    for sample in train_data:
        text = sample['text']
        text_len = len(text)
        
        # 金标正例
        gold_spans = [(arg['start'], arg['end']) for arg in sample.get('arguments', [])]
        positive_count = len(gold_spans)
        total_positive += positive_count
        
        # 目标词区间
        target_start, target_end = sample['target'][0], sample['target'][1]
        target_pairs = [(target_start, target_end)]
        
        # 1. 部署匹配负例：用 task2 预测的 span（非金标）
        sentence_id = sample.get('sentence_id', str(sample['id']))
        predicted_spans = task2_spans.get(sentence_id, [])
        deployment_negatives = [span for span in predicted_spans if span not in gold_spans]
        
        # 2. ±1 边界漂移负例
        drift_negatives = []
        for span in gold_spans:
            drift_negatives.extend(generate_boundary_drift([span], drift_range=1))
        drift_negatives = [span for span in drift_negatives if span not in gold_spans]
        
        # 3. 跨目标负例
        cross_negatives = generate_cross_target_negatives(train_data, target_pairs)
        cross_negatives = [span for span in cross_negatives if span not in gold_spans]
        
        # 4. 随机跨度负例
        random_negatives = generate_random_spans(text, num_spans=5)
        random_negatives = [span for span in random_negatives if span not in gold_spans]
        
        # 合并所有负例
        all_negatives = (
            deployment_negatives[:8] +  # 最多 8 个部署匹配负例
            drift_negatives[:4] +         # 最多 4 个边界漂移
            cross_negatives[:2] +         # 最多 2 个跨目标
            random_negatives[:2]          # 最多 2 个随机
        )
        
        # 去重并限制数量
        all_negatives = list(set(all_negatives))[:int(max_negatives_per_sentence * negative_budget_ratio)]
        
        # 统计
        negative_stats['deployment_match'] += len(deployment_negatives[:8])
        negative_stats['boundary_drift'] += len(drift_negatives[:4])
        negative_stats['cross_target'] += len(cross_negatives[:2])
        negative_stats['random_span'] += len(random_negatives[:2])
        total_negative += len(all_negatives)
        
        # 添加到样本
        sample['negative_spans'] = all_negatives
        sample['negative_count'] = len(all_negatives)
    
    # 保存
    print(f"\n负例挖掘统计:")
    print(f"  部署匹配负例：{negative_stats['deployment_match']}")
    print(f"  边界漂移负例：{negative_stats['boundary_drift']}")
    print(f"  跨目标负例：{negative_stats['cross_target']}")
    print(f"  随机跨度负例：{negative_stats['random_span']}")
    print(f"  总正例数：{total_positive}")
    print(f"  总负例数：{total_negative}")
    print(f"  负正比：{total_negative / max(total_positive, 1):.2f}x")
    
    print(f"\n保存负例增强数据到：{output_file}")
    save_json(train_data, output_file)
    
    print(f"✅ 完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default='./dataset/cfn-dev.json')
    parser.add_argument('--task2_result', type=str, default='./dataset/dev_task2_test.json')
    parser.add_argument('--output_file', type=str, default='./dataset/cfn-dev-null-enhanced.json')
    parser.add_argument('--negative_budget_ratio', type=float, default=2.0)
    parser.add_argument('--max_negatives_per_sentence', type=int, default=16)
    
    args = parser.parse_args()
    
    mine_negative_examples(
        train_file=args.train_file,
        task2_result_file=args.task2_result,
        output_file=args.output_file,
        negative_budget_ratio=args.negative_budget_ratio,
        max_negatives_per_sentence=args.max_negatives_per_sentence
    )