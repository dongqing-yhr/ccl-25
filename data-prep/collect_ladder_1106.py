# -*- coding: utf-8 -*-
"""三资源对照阶梯 11 次训练的读数收集（11-06 发车，任务 #163）。

读数口径写死在这里，避免三份资源各用一把尺子：
  任务三 = 训练日志里 eval() 打印的"金标跨度角色准确率"（每个跨度取 softmax 后第一名，
           与标注逐行相等算对；空类那一列照常计入），取全部轮次里的最高值，也就是选权重那一行的值。
  任务二 = 训练日志里 "new best f1" 那行的跨度层面 F1（阈值取该档命令行传的值），同样取最高值。
  部署口径的角色识别 F1 需要任务二在验证集上给出的跨度清单当候选集，本次 11 次训练都没传
  --dev_cand_file，所以训练日志里没有这个数字；要它得出包后再跑解码侧算分，不在本脚本范围。

判完成只认 saves/ 下的 _best.bin 文件与修改时间；日志文字只用来解释这一格为什么还没出。
两台机器挂同一条 NFS，所以只在一台上跑本脚本就能看到全部 15 份产物，进程存活另查。
"""

import os
import re
import sys
import time

WORK = "/work/workspace/ccl25"

# (资源, 工作副本, 槽位名, 权重前缀, 日志相对路径, 任务号, 这一格在量什么)
ARMS = [
    ("赛题 2.2", "baseline", "R32", "t3_c22_L0", "logfile/t3_c22_L0_1106.log", 3,
     "赛题裸基线：无部署匹配负例、无框架名输入、训练期先验关"),
    ("CFN 1.0", "baseline_c10", "R34-1", "t3_c10_L0", "logfile/t3_c10_L0.log", 3,
     "1.0 裸基线，同上一格"),
    ("CFN 1.0", "baseline_c10", "前置", "t2_c10", "logfile/t2_c10.log", 2,
     "1.0 自己的任务二成员，只用来给负例清单提供误报出处"),
    ("CFN 1.0", "baseline_c10", "R34-2", "t3_c10_L1", "logfile/t3_c10_L1.log", 3,
     "1.0 裸基线只多一份部署匹配负例"),
    ("CFN 1.0", "baseline_c10", "R44-1", "t3_c10_FN", "logfile/t3_c10_FN.log", 3,
     "与 R34-2 逐字同配方，只差框架名进输入这一个开关"),
    ("CFN 1.0", "baseline_c10", "R34-3", "t3_c10_L3", "logfile/t3_c10_L3.log", 3,
     "R34-2 加训练期框架条件先验（tau 0.5／mode frame／alpha 1.0／上限 2.0）"),
    ("英文 1.7", "baseline_en", "R35-1", "t3_en_L0", "logfile/t3_en_L0.log", 3,
     "英文裸基线，同 1.0 的 R34-1"),
    ("英文 1.7", "baseline_en", "前置", "t2_en", "logfile/t2_en.log", 2,
     "英文自己的任务二成员，只用来给负例清单提供误报出处"),
    ("英文 1.7", "baseline_en", "R35-2", "t3_en_L1", "logfile/t3_en_L1.log", 3,
     "英文裸基线只多一份部署匹配负例"),
    ("英文 1.7", "baseline_en", "R44-2", "t3_en_FN", "logfile/t3_en_FN.log", 3,
     "与 R35-2 逐字同配方，只差框架名进输入这一个开关"),
    ("英文 1.7", "baseline_en", "R35-3", "t3_en_L3", "logfile/t3_en_L3.log", 3,
     "R35-2 加训练期框架条件先验（tau 0.5／mode frame／alpha 1.0／上限 2.0）"),
]

CHAINS = [("赛题 2.2", "chain_c22_L0.log", "R32 完成"), ("CFN 1.0", "chain_c10.log", "链 B 全部完成"),
          ("英文 1.7", "chain_en_1106.log", "链 A 全部完成")]

RE_T3_EPOCH = re.compile(r"^epoch (\d+): acc ([0-9.]+)")
RE_T3_BEST = re.compile(r"new best acc ([0-9.]+)")
RE_T3_CUR = re.compile(r"current acc: ([0-9.]+)")
RE_T2_BEST = re.compile(r"new best f1 ([0-9.]+)")
RE_T2_CUR = re.compile(r"current f1: ([0-9.]+)")
RE_T2_EPOCH = re.compile(r"^epoch (\d+):.*?f1 ([0-9.]+)")
RE_DENOM = re.compile(r"验证样本数：(\d+)")
RE_FRAME_HEAD = re.compile(r"dev 帧 top-1 = ([0-9.]+)")
RE_STEPS = re.compile(r"epoch_(\d+) *: *\d+%(?:\|[^|]*)?\| *(\d+)/(\d+) \[(\d+:\d+)<([^,\]]+)")


def read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "rb") as fp:
        raw = fp.read().decode("utf8", "replace")
    return raw.replace("\r", "\n").split("\n")


def pick_reading(lines, task):
    """取这一格最高读数，同时给出该读数出现在第几轮。"""
    if task == 3:
        pairs = [(int(m.group(1)), float(m.group(2))) for m in
                 (RE_T3_EPOCH.match(line) for line in lines) if m]
    else:
        pairs = [(int(m.group(1)), float(m.group(2))) for m in
                 (RE_T2_EPOCH.match(line) for line in lines) if m]
    if not pairs:
        pats = (RE_T3_BEST, RE_T3_CUR) if task == 3 else (RE_T2_BEST, RE_T2_CUR)
        vals = []
        for line in lines:
            for pat in pats:
                m = pat.search(line)
                if m:
                    vals.append(float(m.group(1)))
        return (max(vals), -1) if vals else (None, None)
    epoch, val = max(pairs, key=lambda pr: pr[1])
    return val, epoch


def progress(lines):
    last = None
    for line in lines:
        m = RE_STEPS.search(line)
        if m:
            last = m
    if not last:
        return "（日志里还没有训练进度条，多半在读数据或装权重）"
    return "第 %s 轮 %s/%s 步（已跑 %s，本轮还剩 %s）" % (
        last.group(1), last.group(2), last.group(3), last.group(4), last.group(5))


def main():
    print("[收集] 容器时间 %s（北京时间 +8）" % time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
    print("")
    print("| 资源 | 槽位 | 权重文件 | 修改时间（容器 UTC） | 读数 | 分母 | 口径 | 这一格在量什么 |")
    print("|---|---|---|---|---|---|---|---|")
    n_done = 0
    rows = []
    for res, work, slot, prefix, logrel, task, gloss in ARMS:
        binpath = os.path.join(WORK, work, "saves", prefix + "_best.bin")
        logpath = os.path.join(WORK, work, logrel)
        exists = os.path.exists(binpath)
        nbytes = os.path.getsize(binpath) if exists else 0
        if exists:
            mtime = time.strftime("%m-%d %H:%M", time.gmtime(os.path.getmtime(binpath)))
            size = "%.2f GB" % (nbytes / 1024.0 ** 3)
            # torch.save 往 NFS 上写 1.3~1.8 GB 要几十秒，期间文件已在但字节不全，不能当成跑完
            writing = nbytes < 1.2 * 1024 ** 3
            if writing:
                size += "（正在写）"
                exists = False
        else:
            mtime, size, writing = "未落盘", "-", False
        if exists:
            n_done += 1
        lines = read_lines(logpath)
        val, epoch = pick_reading(lines, task)
        ruler = "任务三金标跨度角色准确率" if task == 3 else "任务二跨度层面 F1"
        reading = "%.4f（第 %s 轮）" % (val * 100, epoch) if val is not None and epoch >= 0 else (
            "%.4f" % (val * 100) if val is not None else "无")
        if exists and not lines:
            reading = "有权重无日志"
        denom = [int(m.group(1)) for m in (RE_DENOM.search(line) for line in lines) if m]
        unit = "标注行" if task == 3 else "验证记录"
        rows.append((res, slot, size, mtime, reading,
                     "%d %s" % (denom[-1], unit) if denom else "无", ruler, gloss, exists))
    for res, slot, size, mtime, reading, denomb, ruler, gloss, exists in rows:
        print("| %s | %s | %s | %s | %s | %s | %s | %s |"
              % (res, slot, size, mtime, reading, denomb, ruler, gloss))
    print("")
    print("[产物] %d/%d 份权重存在；注意训练每轮验证通过都会重写同一份 _best.bin，"
          "存在不等于这一格跑完，跑完只认下面链日志的完成标记" % (n_done, len(ARMS)))
    chain_done = []
    for res, chainlog, done_marker in CHAINS:
        lines = read_lines(os.path.join(WORK, chainlog))
        markers = [line for line in lines
                   if line.startswith("=== [") or (line.startswith("=== R32") and "裸基线" in line)]
        cur = markers[-1] if markers else "（还没进步骤标记）"
        failed = [line for line in lines if "[FAIL]" in line]
        if any(done_marker in line for line in lines):
            state = "已完成"
        elif failed:
            state = "已中断：" + failed[-1][:48]
        else:
            state = "在跑第 %d 步" % len(markers)
        age = int(time.time() - os.path.getmtime(os.path.join(WORK, chainlog))) if os.path.exists(
            os.path.join(WORK, chainlog)) else -1
        chain_done.append(state == "已完成")
        print("[链] %-8s %s（%d 步标记）当前=%s  链日志 %d 秒前写过"
              % (res, state, len(markers), cur[:46], age))
    for res, work, slot, prefix, logrel, task, gloss in ARMS:
        logpath = os.path.join(WORK, work, logrel)
        if not os.path.exists(logpath):
            continue
        age = int(time.time() - os.path.getmtime(logpath))
        if age <= 900:
            print("[在跑] %-10s %-12s 日志 %d 秒前写过  %s"
                  % (res, prefix, age, progress(read_lines(logpath))))
    for res, work, slot, prefix, logrel, task, gloss in ARMS:
        lines = read_lines(os.path.join(WORK, work, logrel))
        heads = [float(m.group(1)) for m in (RE_FRAME_HEAD.search(line) for line in lines) if m]
        if heads:
            print("[帧头] %s %s 任务三模型自带的框架名称头在验证集上的第一名命中 %.4f（最高）"
                  % (work, prefix, max(heads)))
    print("[全部跑完] %s（三条链的完成标记 %d/3）"
          % ("yes" if all(chain_done) else "no", sum(chain_done)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
