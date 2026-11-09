#!/usr/bin/python3
# 从 task2 在训练集上的预测结果里挖出假阳性 span, 作为 task3 NULL 类的部署匹配负例
# 用法: python3 mine_deploy_negatives.py

import argparse
import codecs
import json

parser = argparse.ArgumentParser()
parser.add_argument('--gold_file', default='./dataset/cfn-train.json')
parser.add_argument('--task2_pred', default='./dataset/train_task2_pred.json')
parser.add_argument('--output', default='./dataset/train_deploy_negatives.json')
parser.add_argument('--budget_per_sentence', type=int, default=4,
                    help='每句最多保留几条 FP 负例, 按预测分数排序取前 N')
ARGS = parser.parse_args()


def overlaps(a_s, a_e, b_s, b_e):
    return not (a_e < b_s or b_e < a_s)


with codecs.open(ARGS.gold_file, 'r', encoding='utf8') as f:
    gold = json.load(f)
with codecs.open(ARGS.task2_pred, 'r', encoding='utf8') as f:
    pred = json.load(f)

# sid -> 金标 span 集合 (start,end)
sid2gold = {}
for line in gold:
    sid = line['sentence_id']
    spans = [(s['start'], s['end']) for s in line.get('cfn_spans', [])]
    sid2gold[sid] = spans

# sid -> 该句的所有预测 span (按出现顺序, 后处理可考虑加分数排序)
sid2pred = {}
for item in pred:
    sid, st, en = item[0], item[1], item[2]
    sid2pred.setdefault(sid, []).append((st, en))

negas = []
total_pred = 0
fp_count = 0
for sid, spans in sid2pred.items():
    gold_spans = sid2gold.get(sid, [])
    total_pred += len(spans)
    fps_this = []
    for st, en in spans:
        # 与金标任一 span 有重叠视为 TP
        if any(overlaps(st, en, gs, ge) for gs, ge in gold_spans):
            continue
        fps_this.append((st, en))
    # 截断到 budget
    fps_this = fps_this[:ARGS.budget_per_sentence]
    fp_count += len(fps_this)
    for st, en in fps_this:
        negas.append([sid, st, en])

with codecs.open(ARGS.output, 'w', encoding='utf8') as f:
    json.dump(negas, f, ensure_ascii=False, indent=1)

print('task2 预测 span 总数: {}'.format(total_pred))
print('假阳性 (非重叠): {}'.format(fp_count))
print('每句最多 {} 条上限下, 保留 {} 条'.format(ARGS.budget_per_sentence, len(negas)))
print('输出: {}'.format(ARGS.output))
