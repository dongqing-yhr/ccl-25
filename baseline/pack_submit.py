#!/usr/bin/python3
# 打包某一评测集(dev/A/B)的三个 task 预测结果为 zip, 文件名带分钟级时间戳
# 用法: python3 pack_submit.py --split A

import argparse
import os
import zipfile
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--split", required=True, choices=["dev", "A", "B"])
parser.add_argument("--out_dir", default="./submit")
ARGS = parser.parse_args()

prefix = 'dev_' if ARGS.split == 'dev' else ARGS.split + '_'
src_dir = 'dataset/{}'.format(ARGS.split)
files = ['{}task{}_test.json'.format(prefix, i) for i in (1, 2, 3)]

os.makedirs(ARGS.out_dir, exist_ok=True)
stamp = datetime.now().strftime('%Y%m%d_%H%M')
zip_path = os.path.join(ARGS.out_dir, 'submit_{}_{}.zip'.format(ARGS.split, stamp))

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for name in files:
        src = os.path.join(src_dir, name)
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        # 文件直接放 zip 根目录, 平台不识别嵌套的 submit/ 目录
        zf.write(src, arcname=name)

print('打包完成:', zip_path)
