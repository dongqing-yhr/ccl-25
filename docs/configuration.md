# Frozen configuration and test-track evidence

This page identifies the configuration behind the official CCL 2025 Test B server response
reported in the paper. Development-split ablations are separate measurements; they are not
mixed into this system-level row.

## Training and selection

- Every member uses the released training split of its resource only. Training runs for 10
  epochs with AdamW, learning rate `2e-5` and batch size 8.
- The CFN 2.2 frame ensemble includes seeds 42 and 43. Role members vary training
  settings rather than seeds; the three members launched before 2026-10-23
  (`t1ctl`, `t1neg`, `t1arg`) predate the explicit-seed rule and were started without
  `--seed`, so their exact random draws are not bit-reproducible; the launch commands
  are in `runbash/`.
- Frame-prior strengths are swept on `0..1` and role-prior strengths on `0..2`, both in
  steps of `0.05`. A development-split maximum is selected and reported on that same split.
- The final Test B assembly is a frozen configuration, not the development grid maximum.
  In particular, its CFN 2.2 frame prior uses strength `0.30`.

## Final CFN 2.2 assembly

| stage | frozen configuration | acceptance rule |
|---|---|---|
| Frame identification | Four MacBERT/RoBERTa-large members, including seeds 42 and 43; equal-weight probability average; target-word frame prior at decoding with strength `0.30` | Highest-scoring frame after correction |
| Argument identification | Main and auxiliary RoBERTa-large global-pointer-style detectors | Span union at member cutoffs `-0.25` and `-0.75` |
| Role identification | Six MacBERT-large members, all keyed on out-of-fold frame predictions; they vary the negative pool (4,875 deployment-matched or 27,408 wider), the training-time prior (one member), the frame-name input (five members on, one off), and auxiliary losses (argument pair loss 0.30 in one member, frame auxiliary objective in one member) | Equal-weight probability average; emit when `P(NULL) <= 0.10`; withhold when the highest-scoring role is illegal for the predicted frame |

The deployed frame-conditional role prior is the clamped log form with strength `0.50`,
smoothing `alpha=1.00` and clamp `2.00`. The training-time prior arm is a placement control,
not an additional member inserted into every comparison.

## Official Test B response for this configuration

| FI accuracy | AI F1 | RI F1 | Weighted final |
|---:|---:|---:|---:|
| 73.2188 | 86.3513 | 60.5840 | 72.1047 |

The weighted final uses the published `0.3:0.3:0.4` weighting. The paper prints these values
rounded to two decimals. The baseline and participating-system comparison scores come from
the task overview, not from this server response.

## Related protocols

- English data preparation covers frame elements annotated on the target word itself: the target
  span is a candidate and receives the modal target-role under the predicted frame. This is a
  prerequisite for reading the annotation, not a method variant or a score-gain arm.
- The fixed-pool agreement study uses six members on CFN 2.2 and four on CFN 1.0 and English;
  its acceptance thresholds are `k=5`, `3` and `3`, respectively. These are ablation settings.
- English ablations drop records over 470 characters. The published-parser comparison uses
  the unfiltered released FrameNet 1.7 splits and the parser's evaluator, with the cutoff
  fixed on development before one test read.
- Commands and anchor checks for the English target-admitted curve are in
  [reproduction.md](reproduction.md).
