# The two rulers, and the numbers that pin them

Every reading in the paper and every reading printed by a script in this repository comes from one
function: `score_ri` / `score_ai` / `score_fi` in `baseline/score_official.py`. The copies in
`baseline_c10/` and `baseline_en/` are identical to it (verified file-by-file on the committed
blobs), so the three
resources are measured by identical arithmetic. `--selfcheck` feeds gold back as the prediction and
must print 100.0000 for all three sub-scores; that is the check to run before trusting anything else
on this page.

## Which object each sub-score is computed over

| sub-score | unit it divides by | what it compares |
|---|---|---|
| FI, task 1 frame name | rows of the gold file, **not** deduplicated — the CFN 2.2 development file has 2 300 rows carrying 1 995 distinct `sentence_id` values | one predicted frame name per row, matched against that row's gold frame name, position by position |
| AI, task 2 argument span | sentences (1 995 for CFN 2.2 development), summed over characters | the set of character positions covered by gold spans vs. covered by emitted spans |
| RI, task 3 role name | sentences (1 995 for CFN 2.2 development), summed over triples | the set of `(start, end, role)` triples, boundaries taken exactly |

Two consequences that are easy to get wrong:

**FI and AI/RI have different denominators, so never print them against one number.** `score_fi`
walks `gold_data` by index and compares `pred1[i]` with `gold_data[i]` ([baseline/score_official.py:64](../baseline/score_official.py)). It
counts a row as correct only when the `sentence_id` also matches, and prints a mismatch counter that
has to be 0 — a non-zero counter means the prediction file's row order drifted from the gold file, and
the FI on that run is meaningless. `build_gold` collapses the same gold rows into one entry per
`sentence_id` ([baseline/score_official.py:49](../baseline/score_official.py)), which is the object AI and RI are summed over. So a CFN 2.2 development
measurement is "2 300 rows" for FI and "1 995 sentences" for AI and RI at the same time.
`data-prep/census_splits_1106.py` prints both counts (`rec=` and `uniq_sent=`) for every split of all
three resources; use those, not one number borrowed from another resource.

**Multiple rows for one sentence are merged by union, not by concatenation.** A sentence with two
target words contributes several gold rows; `build_gold` unions their frame names into one set, their
character covers into one set, and their triples into one set. Two rows of the same sentence emitting
the same triple therefore cost nothing extra under RI — the ruler is a set comparison, not a row
comparison.

## The arithmetic, written out

```
char_cover(start, end) = set(range(start, end + 1))      # both ends inclusive: a 1-character span covers 1 character
precision = |intersection| / |prediction set|
recall    = |intersection| / |gold set|
F1        = 2 * precision * recall / (precision + recall)   # 0.0 when precision + recall == 0
```

`char_cover` and `f1` are at [baseline/score_official.py:39](../baseline/score_official.py) and
[baseline/score_official.py:43](../baseline/score_official.py). AI's intersection/precision/recall are character counts
(`len(pc & gc)`, `len(pc)`, `len(gc)`); RI's are triple counts (`len(pp & gg)`, `len(pp)`, `len(gg)`).
A span with correct boundaries and the wrong role name is therefore one false positive and one false
negative under RI at the same time — that is why RI moves less than AI when a decode change re-rolls
role names without touching boundaries.

`score_official.py` also prints a weighted total `0.3*FI + 0.3*AI + 0.4*RI`, because the file is the
official scorer and that formula is the one published in the competition overview. It is printed and
nothing more: no claim in the paper rests on the weighted total, and the three sub-scores are
reported separately for the three resources (the English resource has no weighted total at all).

## Coordinates

`cfn_spans` indices in the gold files are character indices into `list(text)` — the same coordinate
space the predictions are written in — so the scorer compares them directly. This was established by
measurement, not by assumption: the gold indices run past the end of the `word` list, which rules out
word indexing. If a corpus really does ship word-level indices, pass `--expand_words` and
`span_to_chars` flattens through `line["word"][i]["start"] / ["end"]`.

## Frozen anchors

A reading computed off-anchor is not the reading the tables report. These are asserted by code, so
running the assertion is the way to check an environment, not eyeballing a log:

| anchor | value | asserted at |
|---|---|---|
| CFN 2.2 development, task-3 rows emitted by the frozen recipe | 5 221 | [baseline/runbash/grid_member_search.py:270](../baseline/runbash/grid_member_search.py) |
| CFN 2.2 development, rows whose `(start, end, role)` triple matches gold | 3 161 | `--expect_correct` in [baseline/runbash/r48_pair_fuse_1108.py:44](../baseline/runbash/r48_pair_fuse_1108.py) |
| CFN 2.2 development RI for the two-member-plus-rules recipe | 56.8781 | [baseline/runbash/grid_member_search.py:272](../baseline/runbash/grid_member_search.py) (`abs(diff) > 0.01` aborts) |
| CFN 2.2 development AI, task-2 span union at thresholds `-0.25` / `-0.75` over the two members | 88.9700 (6 801 spans) | `--expect_ai` in [baseline/runbash/r38_task2_operator_1108.sh:32](../baseline/runbash/r38_task2_operator_1108.sh) |

The recipe that reproduces the 5 221 / 3 161 / 56.8781 cell is spelled out at
[baseline/runbash/decode_split.py:32](../baseline/runbash/decode_split.py): `--role fn:0.5,la:0.5
--panel fn:0.5,la:0.5 --panel_weight 1.0` with no `--head`, which makes the abstain score the NULL
column of the plain two-member average. `PYTHONPATH=. python3 baseline/runbash/grid_member_search.py
--selfcheck` decodes the already-stored two-member-average probability table as if it were one member
at abstain line 0.10 and aborts if either number moves ([baseline/runbash/grid_member_search.py:29](../baseline/runbash/grid_member_search.py)).
Run it before reading any other row of that script's output. The same selfcheck pass also prints the
illegal-role rescue point (`rescue_min` 0.92, recorded RI 57.4662); that one warns instead of
aborting, because the 5 221 / 56.8781 assertion is what certifies the decode rules.

Two changes silently invalidate the anchor, and the scripts say so rather than printing a wrong
number: switching `--op` to `support` (the three-point renormalising merge, [baseline/runbash/decode_split.py:77](../baseline/runbash/decode_split.py)),
and changing the frame-legal role list the emitted role names are filtered through. `--vote` members
may be added without losing the anchor as long as `--role_rule` stays `avg` and `--vote_min` stays 1.

## How the three resources stay comparable

Each resource is trained on its own released training split and measured on its own development
split, which is the split that carries gold answers; the released test splits have no public answers,
so they cannot be used for a controlled comparison. That is what lets the comparison dimensions line
up: spans always come from that resource's task-2 model, frame names always come from that resource's
task-1 model, one abstain rule and one scorer per resource, and no gold field is read by any decode
step. The one asymmetry to keep in sight: the English FrameNet 1.7 test split does carry gold, and is
therefore reported as a single-resource extra row, never side by side with the other two.
