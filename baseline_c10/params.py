#!/usr/bin/python3

import argparse
import os


def construct_hyper_param():
    parser = argparse.ArgumentParser()
    # train settings
    parser.add_argument('--num_train_epochs', default=10, type=int)
    parser.add_argument('--warmup_proportion', default=0.1, type=float)
    parser.add_argument("--batch_size", default=3, type=int,
                        help="Batch size")
    parser.add_argument("--update_freq", default=1, type=int,
                        help="update_freq")
    parser.add_argument("--accumulate_gradients", default=1, type=int,
                        help="The number of accumulation of backpropagation to effectivly increase the batch size.")
    parser.add_argument('--lr', default=2e-5, type=float, help='model learning rate.')

    # variance-reduction settings
    parser.add_argument('--seed', default=-1, type=int,
                        help="fix RNG seed for reproducible training; -1 keeps the original random behavior")
    parser.add_argument('--with_ema', action='store_true',
                        help="maintain a PostHoc EMA of weights during training and save a separate *_ema.bin checkpoint")
    parser.add_argument('--ema_decay', default=0.999, type=float,
                        help="EMA decay for the shadow weight average")

    # bert settings
    # 底座切换：原基线用 chinese_bert_wwm_ext (12层/768维)，现主用 chinese_roberta_wwm_ext_large (24层/1024维)
    # 权重放共享盘，容器内可见于 /work/models/
    parser.add_argument("--config_file",
                        default='./chinese-roberta-wwm-ext-large/config.json',
                        type=str,
                        help="bert_config_file")
    parser.add_argument("--vocab_file",
                        default='./chinese-roberta-wwm-ext-large/vocab.txt',
                        type=str,
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--init_checkpoint",
                        default='./chinese-roberta-wwm-ext-large/pytorch_model.bin',
                        type=str,
                        help="bert_init_checkpoint")

    parser.add_argument("--with_adv_train",
                        action='store_true',
                        help="with_adv_train")

    parser.add_argument("--no_fgm",
                        action='store_true',
                        help="train_task1/train_task2: skip the FGM embedding attack (they run it every step unless this is set)")

    # 跨度头读哪几层隐状态 / 头的 inner_dim (对手段子成员用的两维)
    parser.add_argument("--layer_fusion", default=None, choices=['last', 'avg2', 'avg4'],
                        help="which encoder layers feed the span head: 'last' = last_hidden_state only, "
                             "'avg2' = mean of the last two, 'avg4' = mean of the last four. "
                             "Not passing it keeps each task's own current behaviour "
                             "(task2 = last, task3 = avg4), so existing recipes stay byte-identical")
    parser.add_argument("--inner_dim", default=None, type=int,
                        help="GlobalPointer head inner dim, also the sinusoidal position-embedding dim. "
                             "Not passing it keeps 64 = current; changes the head shape, so it only makes sense "
                             "for a fresh training run (old checkpoints will not load into the dense layer)")
    parser.add_argument("--pos_weight", default=1.0, type=float,
                        help="task2 GlobalPointer loss: multiply every positive (start,end) pair term by this weight. "
                             "This is the analogue of the opponent's BCE pos_weight, NOT an equivalent operator "
                             "(our loss is the exp-based GlobalPointer multiplicative form, not binary cross entropy). "
                             "1.0 = disabled, byte-identical to current")
    parser.add_argument("--noise_lambda", default=0.15, type=float,
                        help="NoisyTune: scale of the random perturbation added to every pretrained weight before "
                             "training starts. 0.15 is the value hardcoded in train_task1.py/train_task2.py since "
                             "the baseline, so not passing it keeps the current runs byte-identical. 0 disables it.")

    parser.add_argument("--expand_targets", default=0, type=int, choices=[0, 1, 2, 3],
                        help="task2 TRAIN-SET ONLY label augmentation, our own definition of the opponent's "
                             "expand_targets switch (they published the table value but never the operator). "
                             "1 = also mark the variants grown by one character to the left, to the right and to "
                             "both sides; a variant is dropped when it would reach [CLS]/[SEP], touch either of "
                             "the two tokens wrapping the target word, or duplicate an existing span. "
                             "2 = REPLACE every span set by the one where touching/nesting spans are merged into "
                             "a single wider span (shape picked by the fingerprint scan: the opponent emits fewer "
                             "and wider spans than gold); lossless on the character-overlap ruler, loses the inner "
                             "span of a nest on a strict span-set ruler. "
                             "3 = keep every original span and ADD those merged rows. "
                             "0 keeps every training label byte-identical to current. "
                             "The dev set is never grown, so dev F1 stays comparable across the settings.")

    parser.add_argument("--data_dir",
                        default='./data', type=str,
                        help="save_dir")

    parser.add_argument("--save_dir",
                        default='./models', type=str,
                        help="save_dir")

    # submission / prediction settings
    parser.add_argument("--test_file",
                        default='./dataset/cfn-test-A.json', type=str,
                        help="test set to predict on (A榜: cfn-test-A.json, B榜: cfn-test-B.json)")
    parser.add_argument("--task2_result",
                        default='./dataset/A_task2_test.json', type=str,
                        help="task2 prediction result fed into task3")
    parser.add_argument("--task1_result",
                        default='./dataset/A_task1_test.json', type=str,
                        help="task1 prediction result (predicted frame per sentence), used to inject the frame name into task3 input")
    parser.add_argument("--with_frame_name",
                        action='store_true',
                        help="inject the frame name as extra input tokens in task3: gold frame when training, predicted frame when inferring")
    parser.add_argument("--frame_topk",
                        default=1, type=int,
                        help="task3 frame-name injection width: 1 = top-1 frame only (the existing behaviour, byte-identical inputs); "
                             "2 = append the runner-up frame after a second [SEP] as well, so the encoder can see that task1 was unsure "
                             "and pick the role consistent with either candidate. Requires the frame file to carry [sid, f1, p1, f2, p2]")
    parser.add_argument("--model_path",
                        default='', type=str,
                        help="checkpoint to load for prediction; empty means each predict_taskN uses its own default saves/model_taskN_best.bin")
    parser.add_argument("--logit_threshold",
                        default=0.0, type=float,
                        help="task2 decode threshold: a span (i,j) is kept when its logit >= this value (baseline hard-codes 0)")
    parser.add_argument("--span_dump",
                        default='', type=str,
                        help="task2 decode: also write [sentence_id, start, end, span_head_logit] for every emitted span "
                             "to this path. Decode once at a low --logit_threshold and any higher threshold is recoverable "
                             "offline, so the fusion grid costs no extra GPU passes")
    parser.add_argument("--with_nms",
                        action='store_true',
                        help="task2 decode: run NMS over candidate spans, suppressing spans with IoU above --nms_iou against higher-scored spans")
    parser.add_argument("--nms_iou",
                        default=0.5, type=float,
                        help="IoU threshold for task2 NMS; spans overlapping a kept span above this value are dropped")
    parser.add_argument("--refine_boundary",
                        default=0, type=int,
                        help="task2 decode: re-pick each kept span's (start,end) inside a +/-R token neighbourhood "
                             "using the span head's own logit; 0 disables. Never crosses the target-word markers")
    parser.add_argument("--refine_min_gain",
                        default=0.0, type=float,
                        help="task2 refine: move a boundary only when the neighbour logit beats the current one by this margin")
    parser.add_argument("--null_threshold",
                        default=-1.0, type=float,
                        help="task3 NULL abstain rule: <0 keeps argmax behavior (drop when NULL is top-1); >=0 drops a span only when p_null > this value (higher = less abstention = more recall)")
    parser.add_argument("--sid_mode", default='last', choices=['last', 'near'],
                        help="predict_task3_rows.py only: which target-word row of a sentence scores a given span. "
                             "'last' = V0 keeps writing only the sentence's last row (that is what predict_task3.py does, "
                             "so V0/V1/V2 output files stay byte-identical and this switch adds nothing to the pipeline). "
                             "'near' additionally decodes a V3 file where each span is scored by its nearest target word "
                             "(character gap, ties go to the smaller row index) - the literal reading of the opponent's "
                             "--sid-mode near. The opponent never published how they measure 'nearest', how ties break, "
                             "or which row the frame name follows, so V3 is our own reading, not a reproduction of their cell.")
    parser.add_argument("--save_logits",
                        action='store_true',
                        help="write full softmax per span to a sidecar JSON for offline score-level fusion")
    parser.add_argument("--logits_out",
                        default=None, type=str,
                        help="sidecar output path; default dataset/{split}/{prefix}task3_logits.json")
    parser.add_argument("--second_role_threshold",
                        default=-1.0, type=float,
                        help="emit a second role for the same span when its FE prob >= this value; <0 disables")

    # soft cascade: task3 decode restricted to candidate roles chosen by task1 confidence
    parser.add_argument("--soft_cascade",
                        action='store_true',
                        help="task3 decode: mask role logits to the candidate role set of the predicted frame(s); confidence from --task1_conf")
    parser.add_argument("--sc_high",
                        default=0.8, type=float,
                        help="task1 confidence >= sc_high: candidates = roles of the top1 frame")
    parser.add_argument("--sc_low",
                        default=0.5, type=float,
                        help="sc_low <= confidence < sc_high: candidates = union of top1 and top2 frame roles; below sc_low: no mask (global roles)")
    parser.add_argument("--task1_conf",
                        default='./dataset/A_task1_conf.json',
                        type=str,
                        help="task1 confidence file written by predict_task1.py: [sentence_id, top1_frame, top1_prob, top2_frame, top2_prob]")
    parser.add_argument("--frame_roles",
                        default='./dataset/frame_roles.json',
                        type=str,
                        help="frame -> candidate role names table built from the train set")

    # NULL 机制参数
    parser.add_argument("--with_null",
                        action='store_true',
                        help="enable NULL class for abstention mechanism")
    parser.add_argument("--null_weight",
                        default=0.5, type=float,
                        help="NULL class loss weight (default: 0.5)")
    parser.add_argument("--neg_boost",
                        default=1.0, type=float,
                        help="Multiplier applied to the cross-entropy of NULL-labelled (non-argument) "
                             "spans in task3's role loss. Positive spans and the denominator are left "
                             "untouched, so this only amplifies the negatives. 1.0 disables it and keeps "
                             "the historical code path byte-for-byte. This is the analogue of the "
                             "opponent's neg_boost 3/5, NOT an equivalent operator: their base weight is "
                             "unknown, ours is a plain multiplier on top of unchanged positive terms.")

    # logit adjustment: 长尾角色分类的训练期先验修正 (召回侧, 只改 loss 不改推理)
    parser.add_argument("--la_tau",
                        default=0.0, type=float,
                        help="logit adjustment strength; 0 disables it. Training adds tau*log(prior) to role logits before CE; inference uses raw logits")
    parser.add_argument("--la_mode",
                        default='frame', choices=['global', 'frame'], type=str,
                        help="prior source: 'global'=per-role frequency; 'frame'=per-(frame,role) conditional frequency (frame-conditional)")
    parser.add_argument("--la_alpha",
                        default=1.0, type=float,
                        help="smoothing pseudo-count added on top of the global prior when la_mode=frame, so unseen (frame,role) falls back to global")
    parser.add_argument("--la_clamp",
                        default=0.0, type=float,
                        help="floor the logit-adjustment magnitude: clip log-prior to [-la_clamp, 0]; 0 disables clipping (raw can reach -27 and wreck training)")

    # OOF 帧预测 / 部署匹配负例 (跨任务闭环训练)
    parser.add_argument("--train_file",
                        default='./dataset/cfn-train.json', type=str,
                        help="training set path (used by train_task*.py, override for OOF folds)")
    parser.add_argument("--val_file",
                        default='./dataset/cfn-dev.json', type=str,
                        help="validation set path")
    parser.add_argument("--model_prefix",
                        default='model_task1', type=str,
                        help="checkpoint name prefix, saves to saves/{model_prefix}_best.bin")
    parser.add_argument("--oof_frame_file",
                        default=None, type=str,
                        help="OOF 帧预测文件, 若指定则 dataset 用预测帧替代金标帧 (格式: [[sid, frame], ...])")
    parser.add_argument("--deploy_negatives_file",
                        default=None, type=str,
                        help="部署匹配负例文件, 若指定则 task3 NULL 训练时把这些 span 作 NULL 类负例加入")
    parser.add_argument("--out_name",
                        default=None, type=str,
                        help="显式指定预测输出文件路径 (OOF 场景每折写到 dataset/folds/oof_fK.json)")
    parser.add_argument("--prob_dump",
                        default='', type=str,
                        help="task1: path of an .npz holding the FULL softmax over all frame labels for every row "
                             "(probs=N x num_labels float32, sentence_id, idx2label). '' disables it and leaves "
                             "task1_test.json / task1_conf.json byte-identical. Needed to reproduce the top-1..top-20 "
                             "coverage table; the sidecar task1_conf.json only carries top-1/top-2, so it cannot.")

    # 部署口径的选权重集 (task3): 每行 [sid, start, end] 用 range 模型真正会交付的跨度, 不是金标跨度
    parser.add_argument("--dev_cand_file",
                        default='', type=str,
                        help="task3 training: file of task2-proposed dev spans [[sid,start,end],...]; when set, checkpoints are "
                             "selected by deployment-matched micro-F1 on these spans instead of by gold-span accuracy "
                             "(accuracy-best is still kept as saves/{model_prefix}_bestacc.bin)")
    parser.add_argument("--dev_frame_file",
                        default='', type=str,
                        help="task3 training: [[sid, frame], ...] used as the frame-name input of --dev_cand_file "
                             "(deployment-matched predicted frame); empty falls back to the gold frame in --val_file")
    parser.add_argument("--arghood_weight",
                        default=0.0, type=float,
                        help="task3 training: auxiliary argument-hood loss weight. Adds a binary supervision that pushes "
                             "max-over-roles logit against the NULL logit (1 = span is some role, 0 = span is NULL); "
                             "0 disables it")

    # L2 共享编码器多任务: 同一个编码器上再挂一个帧分类头, 总损失 = 角色损失 + lambda * 帧交叉熵
    parser.add_argument("--mtl_frame_weight",
                        default=0.0, type=float,
                        help="task3 training: weight lambda of the shared-encoder frame head (713-class cross entropy on the "
                             "target-word pooled hidden, added to the role loss). 0 = head is not even created, so the run is "
                             "byte-identical to the existing recipes; >0 lets task3 learn the frame-identification process "
                             "inside its own encoder instead of only reading task1's answer. Frame loss is deduplicated per "
                             "sentence inside a batch so long sentences do not get extra weight")

    # R2 task1 帧定义条件化: 定义分支的权重与投影维数 (0 = 连参数都不创建, 与现役逐字节一致)
    parser.add_argument('--def_weight', default=0.0, type=float,
                        help="task1 training: weight of the frame-definition branch added to the "
                             "span Q x K logits. 0 disables it completely (no extra parameters, "
                             "byte-identical forward); >0 needs model_task1_def")
    parser.add_argument('--def_dim', default=256, type=int,
                        help='projection dim of the definition-branch query/key pair')

    args = parser.parse_args()

    # 输出按评测集分目录: dataset/dev|A|B/, 文件前缀 dev_|A_|B_; 未显式覆盖的级联输入跟着切
    base = os.path.basename(args.test_file)
    if 'dev' in base:
        split, prefix = 'dev', 'dev_'
    elif 'test-B' in base:
        split, prefix = 'B', 'B_'
    else:
        split, prefix = 'A', 'A_'
    args.out_dir = './dataset/{}'.format(split)
    args.out_prefix = prefix
    os.makedirs(args.out_dir, exist_ok=True)
    if args.task2_result == './dataset/A_task2_test.json':
        args.task2_result = '{}/{}task2_test.json'.format(args.out_dir, prefix)
    if args.task1_result == './dataset/A_task1_test.json':
        args.task1_result = '{}/{}task1_test.json'.format(args.out_dir, prefix)
    if args.task1_conf == './dataset/A_task1_conf.json':
        args.task1_conf = '{}/{}task1_conf.json'.format(args.out_dir, prefix)

    return args



args = construct_hyper_param()

