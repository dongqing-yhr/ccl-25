#!/usr/bin/python3
"""
task3 NULL 弃权机制训练脚本

用法：
    python3 train_task3_null.py --train_file dataset/cfn-dev-null-enhanced.json
"""

import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizer
from model_task3 import Model
from params import args
import numpy as np
from functools import partial

# 启用 NULL 机制
args.with_null = True
args.null_weight = 0.5  # NULL 损失权重

def get_model_input(data, device=None):
    """
    修改后的输入处理，支持 negative_spans
    """
    def pad(d, max_len, v=0):
        return d + [v] * (max_len - len(d))

    bs = len(data)
    max_len = max([len(x[0]) for x in data])

    input_ids_list = []
    attention_mask_list = []
    target = []
    labels = []
    sentence_id = []

    for d in data:
        input_ids_list.append(pad(d[0], max_len, 0))
        attention_mask_list.append(pad(d[1], max_len, 0))
        target.append(d[2])
        labels.append(d[3])  # 已编码的标签，包含 NULL 类
        sentence_id.append(d[4])

    input_ids = np.array(input_ids_list, dtype=np.compat.long)
    attention_mask = np.array(attention_mask_list, dtype=np.compat.long)

    input_ids = torch.from_numpy(input_ids)
    attention_mask = torch.from_numpy(attention_mask)

    # 构建 H_label（用于计算 loss）
    H_label = torch.zeros(bs, max_len, max_len)
    for i in range(bs):
        for idx in labels[i]:
            if idx[0] < max_len and idx[1] < max_len:
                H_label[i][idx[0], idx[1]] = 1

    return input_ids, attention_mask, target, H_label, sentence_id

def encode_labels_with_null(sample, num_labels, null_label_id):
    """
    编码标签，将 negative_spans 标记为 NULL 类
    
    Args:
        sample: 单个样本（包含 arguments 和 negative_spans）
        num_labels: 正常类别数（不含 NULL）
        null_label_id: NULL 类的 ID（通常是 num_labels）
    
    Returns:
        encoded_labels: 编码后的标签列表
    """
    # 金标正例
    gold_spans = [(arg['start'], arg['end']) for arg in sample.get('arguments', [])]
    
    # 负例
    negative_spans = sample.get('negative_spans', [])
    
    # 编码：正例用正常类别 ID，负例用 NULL 类别 ID
    encoded_labels = []
    for start, end in gold_spans:
        encoded_labels.append((start, end, 0))  # 0 表示正常类
    
    for start, end in negative_spans:
        encoded_labels.append((start, end, null_label_id))  # NULL 类
    
    return encoded_labels

def train_one_epoch(model, train_loader, optimizer, device, epoch, config):
    model.train()
    total_loss = 0.0
    
    for step, batch in enumerate(train_loader):
        input_ids, attention_mask, target, label, sentence_id = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        label = label.to(device)
        target = [(t[0], t[1]) for t in target]  # 保持与原代码一致
        
        optimizer.zero_grad()
        
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            target=target,
            labels=label,
            device=device,
            for_test=False
        )
        
        loss = output['loss']
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids, attention_mask, target, label, sentence_id = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            label = label.to(device)
            target = [(t[0], t[1]) for t in target]
            
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target=target,
                labels=label,
                device=device,
                for_test=True
            )
            
            if output['loss'] is not None:
                total_loss += output['loss'].item()
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")
    
    # 加载配置
    config = BertConfig.from_json_file(args.config_file)
    config.num_labels = 50  # 根据实际角色数调整
    config.with_null = args.with_null
    config.null_weight = args.null_weight
    
    print(f"NULL 机制已启用：{config.with_null}")
    print(f"NULL 损失权重：{config.null_weight}")
    print(f"总类别数（含 NULL）: {config.num_labels + 1}")
    
    # 初始化模型
    model = Model(config)
    model = model.to(device)
    
    # 加载训练数据
    print(f"加载训练数据：{args.train_file}")
    with open(args.train_file, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
    
    # 数据预处理（需要编码标签）
    # 这里简化处理，实际应该用 dataset 类
    print(f"训练样本数：{len(train_data)}")
    
    # 优化器
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr)
    
    # 训练循环
    num_epochs = args.num_train_epochs
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        
        # 训练
        train_loss = train_one_epoch(model, None, optimizer, device, epoch, config)
        print(f"训练损失：{train_loss:.4f}")
        
        # 验证
        val_loss = evaluate(model, None, device)
        print(f"验证损失：{val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            save_path = os.path.join(args.save_dir, 'model_task3_null_best.bin')
            torch.save(model.state_dict(), save_path)
            print(f"保存最佳模型到：{save_path}")
    
    print(f"\n训练完成！最佳验证损失：{best_loss:.4f}")

if __name__ == '__main__':
    main()