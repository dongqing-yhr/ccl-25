#!/usr/bin/env bash
# S4 任务三 8 成员训练驱动（要占卡；发车由用户决定）
# 表 = runbash/s4_members_1031.tsv（配方表未随仓库发布，八名成员取值见论文那一格的表注）。表已于 2026-10-31 按参照材料填全，原先的十二列问号清掉十个；
# 剩两个取值是推断不是原文（null_cls=no 四个成员的 span_thr=none、同四名 neg_boost=1.0），依据写在表头注释里。
# 用法:
#   CHAIN=1 CUDA_VISIBLE_DEVICES=<卡号> bash runbash/s4_task3_members_1031.sh      # b0/rb0/rb2/rb11 串跑
#   CHAIN=2 CUDA_VISIBLE_DEVICES=<卡号> bash runbash/s4_task3_members_1031.sh      # rb6/rb8/rb12/rb19 串跑
#   DRY=1 CHAIN=1 bash runbash/s4_task3_members_1031.sh
#     注意 DRY 的含义与 S1／S2 不同：train_task3.py:325-335 在 CUDA 不可用时 exit(1)，CPU 装配自检走不通，
#     所以 DRY=1 只做"表校验＋底座与清单文件齐备检查＋把将要执行的命令行原样打印"，不启动训练器。
# 会中断发车的门:
#   1) 卡号没给（脚本不猜卡）;  2) saves/ 已有同名权重（不覆盖）;  3) 容器里已有训练/预测进程（共享卡不并发）;
#   4) 表里有问号（一次报全每个成员缺哪几列）;  5) 表里 span_thr／dev_cell 取值不合法、或要用的清单文件缺;
#   6) 底座缺 config.json／vocab.txt／pytorch_model.bin;  7) 训练器退出 0 但没有权重、或日志里没有 dev-RI、
#   或最高 dev-RI 与靶值差超过 ±0.4（±0.4 只当量级对照，靶值口径与我方 dev-RI 未对齐，见表头注释）。
# 串跑不并发: 我方 10-23 实测三臂并发会 OOM（见 .ignor 记录），八成员一律一个跑完再下一个。
set -euo pipefail
cd /work/workspace/ccl25/baseline

TSV=runbash/s4_members_1031.tsv
CHAIN=${CHAIN:-1}
PREFIX_S4=model_task3_s4
TRAIN_DIR=dataset/cmp1031/n9/train
DEV_DIR=dataset/cmp1031/n9/dev
# 表里 dev_cell=baseline 时用生产配方的部署候选清单，不是 N9 生成的两份 cell 清单
PROD_DEV_CAND=dataset/dev/cmp1022/cells/task2_union_m0p25_m0p75.json
# 表里 span_thr=prod 时用生产配方那批部署匹配负例
PROD_TRAIN_NEG=dataset/train_deploy_negatives.json
DEV_FRAMES=./dataset/dev/dev_task1_test.json
LF=logfile
PLAN=dataset/cmp1031/s4/plan_$CHAIN.tsv
mkdir -p "$LF" saves dataset/cmp1031/s4

# 容器里没有 Asia/Shanghai 时区数据（TZ=Asia/Shanghai date 会打出 "Asia 2026"），所以北京时间用 UTC 加 8 小时算。
log() { echo "[$(date -u +%m-%d_%H:%M:%S)Z / $(date -u -d "+8 hours" +%m-%d_%H:%M:%S)BJ] $*"; }
die() { echo "[FATAL] $*" >&2; exit 1; }

[ -s "$TSV" ] || die "缺配方表 $TSV"

if [ "${DRY:-0}" = 1 ]; then
    log "  [dry] 跳过卡号与 GPU 检查, 空 CUDA_VISIBLE_DEVICES 走 CPU 装配自检"
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
else
    [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || die "CUDA_VISIBLE_DEVICES 没设置(卡号由发车人决定, 脚本不猜)"
    nvidia-smi -L >/dev/null 2>&1 || die "容器里看不到 GPU, CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
pgrep -f '[t]rain_task1.py|[t]rain_task2.py|[t]rain_task3.py|[p]redict_task' >/dev/null \
    && die "已有训练/预测进程在跑, 拒绝撞卡" || true

# ---- 门 4/5: 表校验，问号一次报全 ----
python3 - "$TSV" "$CHAIN" "$PLAN" "$TRAIN_DIR" "$DEV_DIR" "$PROD_DEV_CAND" "$PROD_TRAIN_NEG" <<'PY'
import os, sys
tsv, chain, plan, trdir, devdir, prod_dev_cand, prod_train_neg = sys.argv[1:8]
CHAINS = {'1': ['b0', 'rb0', 'rb2', 'rb11'], '2': ['rb6', 'rb8', 'rb12', 'rb19'],
          'all': ['b0', 'rb0', 'rb2', 'rb11', 'rb6', 'rb8', 'rb12', 'rb19']}
if chain not in CHAINS:
    sys.exit('[FATAL] CHAIN 只认 1 / 2 / all，实得 %s' % chain)
want = CHAINS[chain]
rows = [l.rstrip('\n').split('\t') for l in open(tsv, encoding='utf8') if l.strip() and not l.startswith('#')]
hdr = rows[0]
rec = {}
for r in rows[1:]:
    assert len(r) == len(hdr), '表 %s 行宽 %d 与表头 %d 不符' % (r[0], len(r), len(hdr))
    rec[r[0]] = dict(zip(hdr, r))
missing = [m for m in want if m not in rec]
if missing:
    sys.exit('[FATAL] 表里没有这些成员: %s' % ' '.join(missing))
REQ = ['backbone_dir', 'null_cls', 'frame_input', 'layer_fusion', 'inner_dim', 'fgm',
       'span_thr', 'dev_cell', 'lr', 'epochs', 'batch', 'accum', 'seed', 'neg_boost', 'dev_f1']
# 候选清单取值 → 文件名。span_thr 多一个 none（不传 --deploy_negatives_file）与 prod（用生产配方那批部署匹配负例）；
# dev_cell 多一个 baseline（用生产配方的部署候选清单，不在 N9 目录里）。
CELL = {'0.45': 'task2_mean0p45.json', '-0.10': 'task2_meanm0p1.json',
        '-0.1': 'task2_meanm0p1.json', '0.450': 'task2_mean0p45.json'}
bad = []
out = []
for m in want:
    d = rec[m]
    q = [c for c in REQ if d[c].strip() in ('', '?')]
    if q:
        bad.append('    %s 缺 %s' % (m, '、'.join(q)))
        continue
    st, dc = d['span_thr'].strip(), d['dev_cell'].strip()
    # 训练侧 NULL 负例清单
    if st == 'none':
        trfn = ''
    elif st == 'prod':
        trfn = prod_train_neg
        if not os.path.isfile(trfn) or os.path.getsize(trfn) == 0:
            bad.append('    %s 训练侧要用 span_thr=%s，但生产配方的部署匹配负例 %s 不存在或为空'
                       % (m, st, trfn))
            continue
    elif st in CELL:
        trfn = CELL[st]
        pth = os.path.join(trdir, trfn)
        if not os.path.isfile(pth) or os.path.getsize(pth) == 0:
            bad.append('    %s 训练侧要用 span_thr=%s，但清单文件 %s 不存在或为空（N9 的训练侧清单还没生成）'
                       % (m, st, pth))
            continue
    else:
        bad.append('    %s 的 span_thr=%s 只能是 0.45／-0.10／prod／none' % (m, st))
        continue
    # 验证侧选权重清单
    if dc == 'baseline':
        devfn = prod_dev_cand
        if not os.path.isfile(devfn) or os.path.getsize(devfn) == 0:
            bad.append('    %s 验证侧要用 dev_cell=baseline，但生产配方的部署候选清单 %s 不存在或为空' % (m, devfn))
            continue
    elif dc in CELL:
        devfn = os.path.join(devdir, CELL[dc])
        if not os.path.isfile(devfn) or os.path.getsize(devfn) == 0:
            bad.append('    %s 验证侧要用 dev_cell=%s，但清单文件 %s 不存在或为空（先跑 N9）' % (m, dc, devfn))
            continue
    else:
        bad.append('    %s 的 dev_cell=%s 只能是 0.45／-0.10／baseline' % (m, dc))
        continue
    if d['fgm'] not in ('on', 'off'):
        bad.append('    %s 的 fgm=%s 只能是 on／off' % (m, d['fgm']))
    if d['null_cls'] not in ('yes', 'no'):
        bad.append('    %s 的 null_cls=%s 只能是 yes／no' % (m, d['null_cls']))
    if d['frame_input'] not in ('yes', 'no'):
        bad.append('    %s 的 frame_input=%s 只能是 yes／no' % (m, d['frame_input']))
    if d['layer_fusion'] not in ('last', 'avg2', 'avg4'):
        bad.append('    %s 的 layer_fusion=%s 不在我方 --layer_fusion 候选 last／avg2／avg4 里'
                   % (m, d['layer_fusion']))
    try:
        nb = float(d['neg_boost'])
        if nb < 1.0:
            raise ValueError
    except ValueError:
        bad.append('    %s 的 neg_boost=%s 要写成 ≥1 的数字，关闭时写 1.0' % (m, d['neg_boost']))
        continue
    # 训练侧不传负例文件时写字面量 NONE 而不是空字段：IFS=tab 的 read 会把连续制表符并成一个分隔符，
    # 空字段会让后面所有列整体左移一位（10-31 第一次 DRY 就踩在 --dev_cand_file 被传成空值上）。
    out.append('\t'.join([m] + [d[c] for c in REQ] + [trfn or 'NONE', devfn]))
if bad:
    print('[FATAL] 配方表不能发车，问题按成员列出如下（问号＝参照材料里没拿到的取值，不猜；'
          '靠推断得来的取值已在表头注释里标明出处）:')
    for b in bad:
        print(b)
    sys.exit(1)
with open(plan, 'w', encoding='utf8') as fh:
    fh.write('member\t' + '\t'.join(REQ) + '\ttrain_neg_file\tdev_cand_file\n')
    for line in out:
        fh.write(line + '\n')
print('[table] %s 校验通过: %d 个成员，问号 0 格 -> %s' % (tsv, len(out), plan))
for line in out:
    print('        ' + line)
PY

log "卡=${CUDA_VISIBLE_DEVICES:-空} DRY=${DRY:-0} 链=$CHAIN 表=$TSV 计划=$PLAN"

run_member() {   # run_member <member> <backbone_dir> <null_cls> <frame_input> <layer_fusion> <inner_dim> <fgm>
                 # <span_thr> <dev_cell> <lr> <epochs> <batch> <accum> <seed> <neg_boost> <dev_f1>
                 # <train_neg_file> <dev_cand_file>
    local M=$1 BB=$2 NULL=$3 FR=$4 LFU=$5 ID=$6 FGM=$7 ST=$8 DEVC=$9 LR=${10} EP=${11} BS=${12} AC=${13} \
          SD=${14} NB=${15} TGT=${16} TRN=${17} DVN=${18}
    local PREFIX="${PREFIX_S4}_${M}"
    local CKPT="saves/${PREFIX}_best.bin"
    local LOGF="$LF/train_task3_${M}_$(date -u +%m%d_%H%M).log"
    [ -f "$CKPT" ] && die "$CKPT 已存在，换前缀或先归档，不覆盖已有权重"
    for f in config.json vocab.txt pytorch_model.bin; do
        [ -s "$BB/$f" ] || die "成员 $M 的底座 $BB 缺 $f"
    done
    [ -s "$DEV_FRAMES" ] || die "缺 $DEV_FRAMES（选权重时部署候选集要用的框架名称来源）"
    local EXTRA=()
    local TRNARG="NONE(不传 --deploy_negatives_file)"
    [ "$FGM" = off ] && EXTRA+=(--no_fgm)
    [ "$FR" = yes ] && EXTRA+=(--with_frame_name --oof_frame_file ./dataset/train_oof_frames.json)
    if [ "$TRN" != NONE ]; then
        [ -s "$TRN" ] || die "成员 $M 的训练负例清单 $TRN 不存在或为空"
        EXTRA+=(--deploy_negatives_file "$TRN")
        TRNARG="$TRN"
    fi
    [ -s "$DVN" ] || die "成员 $M 的选权重清单 $DVN 不存在或为空"
    if [ "$NULL" = no ]; then
        log "  [WARN] 成员 $M 标的是\"不带 NULL 类\"。我方表达不出 null_cls=no：train_task3.py:19 硬写 with_null=True，"
        log "         :400-401 因此恒把类别数加一（NULL 列永远在），而 config.with_null 从未设置，"
        log "         model_task3.py:33 读到 False、:134 的 NULL 加权分支从不执行。b0／rb0／rb2／rb11 四个成员"
        log "         产出的是\"带 NULL 列但不喂 NULL 负例\"的近似物，交付对照表里标\"近似复现\"，不标\"复现\"（工作流 §16.3）。"
    fi
    log "===== $M 底座=$BB null_cls=$NULL frame_input=$FR fusion=$LFU inner_dim=$ID fgm=$FGM"
    log "      span_thr=$ST dev_cell=$DEVC lr=$LR ep=$EP bs=$BS accum=$AC seed=$SD neg_boost=$NB 靶值=$TGT"
    log "      训练负例=$TRNARG  选权重清单=$DVN  日志=$LOGF"
    # DRY=1 的含义与 S1／S2 不同：train_task3.py:325-335 在没有 CUDA 时直接 exit(1)，CPU 装配自检走不通。
    # 所以 DRY=1 只做到"表校验＋文件齐备＋把将要执行的命令行原样打出来"，不启动训练器。
    if [ "${DRY:-0}" = 1 ]; then
        echo "    [dry] 将要执行: python3 -u train_task3.py --seed $SD --model_prefix $PREFIX" \
             "--config_file $BB/config.json --vocab_file $BB/vocab.txt --init_checkpoint $BB/pytorch_model.bin" \
             "--dev_cand_file $DVN --dev_frame_file $DEV_FRAMES" \
             "--num_train_epochs $EP --batch_size $BS --accumulate_gradients $AC --lr $LR" \
             "--layer_fusion $LFU --inner_dim $ID --neg_boost $NB" ${EXTRA[@]+"${EXTRA[@]}"}
        return 0
    fi
    python3 -u train_task3.py \
        --seed "$SD" --model_prefix "$PREFIX" \
        --config_file "$BB/config.json" --vocab_file "$BB/vocab.txt" --init_checkpoint "$BB/pytorch_model.bin" \
        --dev_cand_file "$DVN" --dev_frame_file "$DEV_FRAMES" \
        --num_train_epochs "$EP" --batch_size "$BS" --accumulate_gradients "$AC" --lr "$LR" \
        --layer_fusion "$LFU" --inner_dim "$ID" --neg_boost "$NB" \
        ${EXTRA[@]+"${EXTRA[@]}"} > "$LOGF" 2>&1
    # ---- 跑完的验收：权重在、日志里有 dev-RI、最高 dev-RI 与靶值差 ≤0.4 ----
    [ -s "$CKPT" ] || die "$M 退出 0 但没有 $CKPT"
    grep -a 'dev-RI' "$LOGF" >/dev/null || die "$M 的日志 $LOGF 里一个 dev-RI 都没打，选权重口径没生效"
    python3 - "$LOGF" "$M" "$TGT" <<'PY'
import re, sys
logf, m, tgt = sys.argv[1], sys.argv[2], float(sys.argv[3])
vals = [float(x) for x in re.findall(r'dev-RI[ =]+([0-9.]+)', open(logf, encoding='utf8').read())]
if not vals:
    sys.exit('[FATAL] %s 日志里解析不出 dev-RI 数字' % m)
best = max(vals)
print('  [验收] %s 训练里最高 dev-RI = %.4f，靶值 %.3f，差 %+.3f（阈值 ±0.4；'
      '注意靶值是参照实现那一档的\"部署口径官方四元组 F1\"，我方 dev-RI 的 span 集合与框架名称来源都和参照实现没对齐，'
      '±0.4 只当量级对照，不能当复现判据）' % (m, best, tgt, best - tgt))
sys.exit(0 if abs(best - tgt) <= 0.4 else 1)
PY
    log "  [OK] $M -> $CKPT"
}

# 主循环留在当前 shell 里跑：写成 `计划 | while read` 的话循环体进子 shell，成员失败时的 die 只退出子 shell，
# 脚本会照样走到最后一行打印 S4_DONE——那就不是 fail loudly 了。
mapfile -t PLAN_ROWS < <(tail -n +2 "$PLAN")
[ "${#PLAN_ROWS[@]}" -gt 0 ] || die "计划 $PLAN 里没有可跑的成员"
for ROW in "${PLAN_ROWS[@]}"; do
    IFS=$'\t' read -r M BB NULL FR LFU ID FGM ST DEVC LR EP BS AC SD NB TGT TRN DVN <<< "$ROW"
    run_member "$M" "$BB" "$NULL" "$FR" "$LFU" "$ID" "$FGM" "$ST" "$DEVC" "$LR" "$EP" "$BS" "$AC" \
               "$SD" "$NB" "$TGT" "$TRN" "$DVN"
done

log "S4 链 $CHAIN 跑完（DRY=${DRY:-0}）。八份概率表接着走: dump 用 runbash/dump_member_probs.py，融合解码用 runbash/s5_vote2_decode_1031.sh"
echo S4_DONE
