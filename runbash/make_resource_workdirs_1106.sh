#!/bin/bash
# 给 CFN 1.0 与英文两份资源各建一个 baseline 工作副本（容器内跑，2026-11-06）。
# 为什么要副本：train_task2.py:295/299 与 train_task3.py:382/389/481 把类别表路径硬编码成
# ./dataset/frame_info.json，换资源必须换表；副本还顺带把 saves/ 分开，两条链不会互相覆盖权重。
# 副本只拷代码（.py、runbash），不拷底座目录与任何已有产物；底座用相对路径 ../baseline/... 指过去。
set -e
W=/work/workspace/ccl25

for D in baseline_c10 baseline_en; do
    if [ -d "$W/$D" ]; then
        echo "[已存在] $D  不重建，只补数据文件"
    else
        mkdir -p "$W/$D/saves" "$W/$D/dataset"
        tar -C "$W/baseline" -cf - \
            --exclude=./saves --exclude=./dataset --exclude=./dumps --exclude=./models \
            --exclude=./submit --exclude=./theirs_dump --exclude=./logfile \
            --exclude=./lert_t1_knobs_td6 --exclude=./t2repro --exclude=./__pycache__ \
            --exclude=./bge-large-en-v1.5 --exclude=./chinese-macbert-large \
            --exclude=./chinese-roberta-wwm-ext-large --exclude=./chinese_bert_wwm_ext . \
            | tar -C "$W/$D" -xf -
        echo "[新建] $D  代码文件 $(ls "$W/$D"/*.py | wc -l) 个 .py"
    fi
    mkdir -p "$W/$D/dataset" "$W/$D/saves"
done

cp "$W/datasets_paper/cfn1.0/frame_info.json"          "$W/baseline_c10/dataset/frame_info.json"
cp "$W/datasets_paper/cfn1.0_task3/task3_train.json"   "$W/baseline_c10/dataset/"
cp "$W/datasets_paper/cfn1.0_task3/task3_dev.json"     "$W/baseline_c10/dataset/"

cp "$W/datasets_paper/framenet_task3/frame_info_en.json" "$W/baseline_en/dataset/frame_info.json"
cp "$W/datasets_paper/framenet_task3/task3_train.json"  "$W/baseline_en/dataset/"
cp "$W/datasets_paper/framenet_task3/task3_dev.json"    "$W/baseline_en/dataset/"
cp "$W/datasets_paper/framenet_task3/task3_test.json"   "$W/baseline_en/dataset/"

echo "=== 代码与主目录逐字节对照（应只有底座/数据目录不同）==="
for F in train_task2.py train_task3.py dataset_task2.py dataset_task3.py model_task2.py model_task3.py params.py; do
    A=$(md5sum "$W/baseline/$F" | cut -d' ' -f1)
    B=$(md5sum "$W/baseline_c10/$F" | cut -d' ' -f1)
    C=$(md5sum "$W/baseline_en/$F" | cut -d' ' -f1)
    echo "$F  主=$A  c10=$B  en=$C  $( [ "$A" = "$B" ] && [ "$A" = "$C" ] && echo 相同 || echo 不一致 )"
done

echo "=== 数据文件落地核对 ==="
md5sum "$W/baseline_c10/dataset/"* "$W/baseline_en/dataset/"*
echo "=== 底座可达性（副本里用 ../baseline/<底座> 相对路径指过去）==="
for P in chinese-macbert-large bge-large-en-v1.5; do
    echo "$P -> $(ls "$W/baseline/$P" 2>/dev/null | tr '\n' ' ')"
done
