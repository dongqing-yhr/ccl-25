# CFSP 2025 frame semantic parsing — reproduction code

Code for the paper under review. Everything needed to rebuild the reported readings from
the released data and the released pretrained encoders is here; the corpora, the
pretrained weights and the trained checkpoints are not redistributed (see
[What is not in this repository](#what-is-not-in-this-repository)).

中文概述：本仓库是按本文方法做实验的全部代码。三段级联的任务一（框架名称）、任务二（论元跨度）、
任务三（框架元素角色名称）的生产代码在 `baseline/`，三套资源（赛题 CFN 2.2、CFN 1.0、英文 FrameNet 1.7）
各自的同构工作目录在 `baseline_c10/`、`baseline_en/`，跨资源对照阶梯的入口在 `runbash/`。语料、预训练权重、
训练出来的检查点不随仓库分发，下载口见下。两把尺子的定义与三条冻结锚点见 `docs/scoring.md`，
几个会让人卡住的坑见 `docs/reproduction.md`。

## What the system does

A three-stage cascade, each stage a separate fine-tuned encoder:

| stage | output | trained by | scored by |
|---|---|---|---|
| task 1 | 框架名称 (frame name) for every row | `train_task1.py` | frame-name accuracy FI |
| task 2 | 论元跨度 (argument spans) as `[start, end]` character pairs | `train_task2.py` | character-coverage F1 AI |
| task 3 | 框架元素角色名称 (role names) for the spans task 2 emitted | `train_task3.py` | strict `(start, end, role)` triple F1 RI |

Task 3 consumes task 2's spans, and in the arm whose input carries the frame name it also consumes
task 1's predicted frame name. Every reading in the paper is produced by running that same cascade
over a split that has gold annotations — the development split is used as if it were a test set with
the answers published, so the comparison dimensions (spans from task 2, frame names from task 1, the
one abstain rule, the one scorer) are identical across the three resources.

## Layout

| path | contents |
|---|---|
| `baseline/` | CFN 2.2 working directory: production training and decoding code, plus `runbash/` with the shared operators and the reading generators |
| `baseline_c10/` | CFN 1.0 working directory, same code tree |
| `baseline_en/` | FrameNet 1.7 working directory, same code tree |
| `runbash/` | cross-resource entry points: the three-ring decode ladder, the GPU queues that drive it, the probability sidecar dump, the equal-strength prior control |
| `data-prep/` | scripts that build the CFN 1.0 and FrameNet 1.7 task-3 records and count each resource's label statistics |
| `optional-backbones/` | adapters for two encoders that are not ours (LERT, Chinese-ModernBERT) used only for the "does the backbone carry the gain" control cell |
| `docs/` | `scoring.md` (the two rulers and the frozen anchors), `reproduction.md` (pitfalls that cost us a run) |

The three working directories are the same code. Measured on the committed blobs: `baseline_c10/` and
`baseline_en/` are identical to each other, file for file (46 files each). Of the 45 paths shared by
all three directories, one differs in content — `runbash/dump_member_probs.py`, which in the two newer
resources gained `--model_config`/`--model_vocab` so one dump call can point every member at a
different pretrained encoder. Eight files ship only in `baseline/`: the two files the competition
distributed (`README.txt`, `requirements.txt`) and six reading generators that were run on the
competition resource only. One file ships only in `baseline_c10/` and `baseline_en/`:
`runbash/probe_gamma_control_1108.py`, the equal-strength prior control on the two newer resources.
`.gitattributes` forces LF endings so a checked-out `*.sh` runs under bash on Linux; the local working
copies that produced these readings are a CRLF/LF mix, which is the only reason two of the shared
files (`dataset_task3.py`, `score_official.py`) looked different before staging.
Directory names are not normalised on purpose: the scripts refer to
`/work/workspace/ccl25/baseline*/...` literally, and renaming the folders would break every chain.

## Environment

`requirements.txt` pins the versions the reported runs used (Python 3.11.13, PyTorch 2.8.0+cu128,
CUDA 12.8, transformers 4.35.2). The upstream baseline's own requirements file is left as it was
distributed and is wrong in two ways — it pins an older torch/transformers pair and lists `codecs`
and `json`, which are standard-library modules, so `pip install -r` on that file fails.

## Data and weights to fetch

| what | where it came from |
|---|---|
| CFN 2.2 (competition data: `cfn-dev.json`, `cfn-test-A.json`, `cfn-test-B.json`) | Tianchi dataset page `https://tianchi.aliyun.com/dataset/149079` |
| CFN 1.0 | `https://github.com/SXUNLP/Chinese-Frame-Semantic-Parsing`, files under `ccl-cfn/`; the path we actually pulled from is the fork mirror `https://raw.githubusercontent.com/masterpen/Chinese-Frame-Semantic-Parsing/HEAD/ccl-cfn/` |
| FrameNet 1.7 splits for English | `https://hf-mirror.com/datasets/liyucheng/FrameNet_v17/resolve/main/` (used by `data-prep/dl_framenet_splits_1105.py`) |
| `hfl/chinese-bert-wwm-ext`, `hfl/chinese-roberta-wwm-ext-large`, `hfl/chinese-macbert-large` | HuggingFace, or the mirror `https://hf-mirror.com` if the hub is unreachable from your network |
| `BAAI/bge-large-en-v1.5` (English encoder for task 3) | `data-prep`-adjacent download script; **read the key-name note in `docs/reproduction.md` before loading this one into task 3** |

CFN is released under CC BY-NC 4.0 (non-commercial research use), so the corpora are not mirrored
here. Place the files under each working directory's `dataset/` and `datasets_paper/` trees.

## Minimal run (one resource)

```bash
cd baseline
python train_task1.py --seed 42 ...   # frame names
python train_task2.py --seed 42 ...   # argument spans
python train_task3.py --seed 42 ...   # roles, reads task 2 output
python predict_task2.py ...           # must run before predict_task3.py
python predict_task3.py ...
python score_official.py --split dev
```

The authoritative ordering note is `baseline/README.txt` (upstream file, kept as distributed).
Always pass `--seed 42`: two runs of one identical recipe move frame-name accuracy by about one
point on their own, which is large enough to hide or fake a methodological difference.

## The controlled cross-resource ladder

```bash
bash runbash/decode_ladder_res_1106.sh c22     # or c10 / en
```

One script, three resources, three rings in cascade order: task 1's frame names, task 2's spans with
the same emit threshold (`-0.5`, the threshold used when the deployment-matched negative list was
mined), task 3 over the spans task 2 emitted with the same abstain rule (no `--null_threshold` passed:
a row whose role argmax lands on the null class is not emitted). Each ring's predictions land in that
resource's own `dataset/devladder_<resource>/` directory, so no existing product is overwritten.

`runbash/start_queues_1106.sh` and `runbash/queue_decode_1106.sh` wrap the same ladder in a GPU queue;
they name a machine as `${CCL25_HOST}` and take the container-visible GPU index as an argument, so set
that variable to your own host before running them.

## Which script produces which reading

| slot in the paper's tables | what it measures | generator |
|---|---|---|
| R34 / R35 / R44 ladder rows | the four task-3 recipe arms per resource, in deployment order | `runbash/decode_ladder_res_1106.sh` |
| R36 | equal-strength prior control: conditional table, class prior, table with the link shuffled | `runbash/probe_gamma_control_1108.py` (CFN 1.0, English), `baseline/runbash/probe_gamma_control_1027.py` (CFN 2.2) |
| R37 | how many rows each member votes for, and abstention by vote count | `*/runbash/fuse_vote_1108.py` |
| R38 | task-2 fusion operator with the members fixed: per-member thresholds unioned, aligned average, unaligned average, plus the identity check that says when aligning is only a threshold shift | `baseline/runbash/r38_task2_operator_1108.sh`, `r38_scale_quantiles_1108.py`, `r38_idcheck_diag_1108.py` |
| R48 | two-member probability average against each member alone | `*/runbash/r48_pair_fuse_1108.py` |
| R50 | prior applied at training time against prior applied at decode time | `*/runbash/r50_decode_prior_1108.py` |
| probability sidecars that R37, R48 and R50 read | per-member probability dumps | `*/runbash/dump_member_probs.py`, `*/runbash/probs_t3_to_npy_1108.py` |

`*` is `baseline`, `baseline_c10` or `baseline_en`. The scripts that take paths as arguments run from
whichever working directory is being measured; the frozen anchors below are what tells you the run is
on the intended ruler.

## Frozen anchors

A reading computed off-anchor is not the reading the table reports. Each generator asserts these
before printing anything else:

- CFN 2.2 development split: 5 221 emitted role rows, 3 161 rows whose `(start, end, role)` triple
  matches gold, RI 56.8781 for the two-member-plus-rules recipe.
- Task-2 span union at thresholds `-0.25` / `-0.75` on the two members: 6 801 spans, AI 88.9700 on
  the 1 995-sentence scorer.
- `bash baseline/runbash/grid_member_search.py` self-check line reproduces the 56.8781 cell
  byte-for-byte before any other row of its output should be read.

## What is not in this repository

- Corpora, pretrained weights, checkpoints (225 zip archives and 393 MB of `pytorch_model.bin` files
  in the working tree), prediction products, run logs, submission packages.
- The paper text. Tables and numbers are in the paper, not here.
- Three drivers that were kept out on purpose: `runbash/s2_task2_arms_1031.sh`,
  `runbash/s5_vote2_decode_1031.sh`, `runbash/t2_shift_ours_vs_theirs_1103.sh`. They hard-code another
  team's unpublished development figures or a copy of that team's hyper-parameter table, and those
  figures are not ours to redistribute. The operators those drivers called
  (`task2_shift_mean.py`, `task2_vote_merge.py`, `n10_longtail_aug_1031.py`) are included, so the
  mechanism stays reproducible; only the baked-in third-party numbers are absent.
- One-off probes, packaging scripts and ledger tooling from the leaderboard phase. Nothing in the
  paper depends on them.

## Reproduction notes worth reading first

`docs/reproduction.md` — the encoder key-name trap that silently trains a random initialisation, the
470-character input cap for English records and where the number comes from, why the legal role list is
unioned across splits, how to tell whether a cell actually finished when every round rewrites the same
checkpoint name.

`docs/scoring.md` — the two rulers written out, which denominator each uses, and the difference between
the 1 995-sentence scorer and the 2 300 released development rows (never print those two as one
denominator).

## License

The code is MIT, in `LICENSE`. Two exceptions, both narrow: `baseline/README.txt` and
`baseline/requirements.txt` are redistributed verbatim from the competition baseline package for
provenance (see pitfall 8 in `docs/reproduction.md`), and the corpora themselves are CC BY-NC 4.0 and
are not mirrored here, so the data-licensing note above governs anything you fetch separately.
