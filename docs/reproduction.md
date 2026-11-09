# Pitfalls that cost us a run

Each item is something that either produced a plausible-looking wrong number or aborted a chain, and
each one has a check that can be run in seconds. Ordered by how much damage they do silently.

## 1. A prediction script with no `--model_path` evaluates the incumbent model

`predict_task1.py:149`, `predict_task2.py:221`, `predict_task3.py:310` and
`predict_task3_rows.py:54` fall back to `saves/model_task{1,2,3}_best.bin` when no checkpoint is
passed. Nothing warns. A controlled comparison that forgets the flag therefore scores the incumbent
weights twice and reports the "new" arm as whatever the incumbent does — which is how two arms can
look identical without anyone touching the same file.

Check before reading a comparison: the predict log prints the checkpoint it loaded, and
`saves/model_taskN_best.bin` is the file the incumbent chain writes. Every controlled run must pass
`--model_path` explicitly with a per-arm name (`--model_prefix`), and the run's own log line naming
the loaded weights is the evidence, not the command line.

## 2. `strict=False` turns a mismatched encoder into a silently random initialisation

`model_task3.py:27` declares the encoder as `self.bert = BertModel(config)`, so every encoder
parameter is named `bert.*`. `train_task3.py:421` loads the initial weights with
`load_state_dict(state, strict=False)`, and `strict=False` does not raise on unmatched keys.

`BAAI/bge-large-en-v1.5` ships its keys **without** the `bert.` prefix. Measured by
`data-prep/smoke_load_arms_1106.py`: 392 keys in the file, 0 of them prefixed, 391 missing on the
model — the whole encoder comes up random-initialised and training still prints normal loss curves.
`hfl/chinese-macbert-large` is prefixed (0 missing), which is why the CFN 2.2 and CFN 1.0 chains were
never affected and only the English chain hit this.

The fix is a renamed copy, not an edit: adding the `bert.` prefix writes
`pytorch_model_bertprefixed.bin` next to the untouched download, and the English chain's
`--init_checkpoint` points at the new file. `embeddings.position_ids` is a read-only buffer registered
by older `BertEmbeddings`; current transformers has no such parameter, so it stays an unexpected key
and is ignored — it is deliberately kept in the file so the two versions remain key-by-key comparable.

Any time an encoder is swapped, run the smoke check first and read the missing/unexpected key counts.
Zero missing is the only acceptable answer.

## 3. Every epoch rewrites the same checkpoint name, so "the file exists" is not "the cell ran"

Weights land at `saves/<prefix>_best.bin` and the decode step overwrites its output slot. A checkpoint
from yesterday satisfies `test -f`. The chains therefore gate on artifacts and timestamps, not on
existence: `runbash/decode_ladder_res_1106.sh:124` refuses to score a task-3 cell whose prediction
file is older than the step's start time (`拒绝拿旧文件算分` — "refusing to score a stale file"), and
the only completion signal is the chain's last line, `=== 解码链 <res> 完成 <time> ===`
([runbash/decode_ladder_res_1106.sh:145](../runbash/decode_ladder_res_1106.sh)). A chain log without that
line means the run is unfinished, whatever is sitting in `saves/`.

The same script backs up a pre-existing task-3 prediction slot before the sweep and restores it by md5
afterwards, so a re-decode cannot silently leave the incumbent's answers in place.

## 4. The label table path is hard-coded, so switching resource means switching table

`train_task2.py:295`, `:299` and `train_task3.py:382`, `:389`, `:481` read `./dataset/frame_info.json`
literally — the label table path has no command-line switch. That is why the other two resources
are separate working directories rather than a `--frame_info` flag: the copy also keeps `saves/`
apart, so two chains cannot overwrite each other's weights. `runbash/make_resource_workdirs_1106.sh`
builds the copies from `baseline/` (code only — it excludes `saves/`, `dataset/`, `dumps/`, `models/`
and every encoder directory, and the encoders are then reached through relative paths back into
`../baseline/`), then drops the resource's own `frame_info.json` into place.

Two further consequences of a new label table:

- **The table must cover every frame name and every role name in all three of that resource's
  splits, or loading dies with a `KeyError`** at `dataset_task2.py:87` (`self.label2cls[d1["frame"]]`)
  and `dataset_task3.py:81` (`self.label2idx[spans["fe_name"]]`). FrameNet 1.7 ships no frame
  declaration file that travels with the splits, so the English table's frame list and role lists are
  the union of what actually occurs in train/dev/test — built by `data-prep/prep_en_task3_1106.py`,
  counted by `data-prep/stat_en_legalrole_1106.py`.
- **Which splits feed which table is a methodological line, not a convenience.** The training-time
  priors and the deployment-matched negatives are built from the training split only; the union over
  all three splits is used solely for the legality lookup that prevents the `KeyError`.

## 5. English records are capped at 470 characters, and the cap is arithmetic

The loader tokenises the raw sentence character by character, and English measures about 1 token per
character (including spaces). The sequence is `len(text) + 4` (`[CLS]`, `[SEP]`, and the two target
markers), plus `1 + len(frame name)` when the frame name is injected — and the injected name is gold
at training time but task 1's prediction at inference time, so the margin must be sized on the longest
frame name in the whole table. For FrameNet 1.7 that name is `Cause_change_of_position_on_a_scale`
(35 characters), giving `512 − 4 − 1 − 35 − 2 = 470` (`MAX_CHARS` in
[data-prep/prep_en_task3_1106.py:26](../data-prep/prep_en_task3_1106.py)).

Records over the cap are dropped, and both counts — dropped and kept, rows and spans — are printed so
the denominator of every English reading is explicit. Do not raise the cap without re-doing that
arithmetic against the resource's own longest frame name.

## 6. `--seed` defaults to `-1`, which means "not reproducible"

`params.py:21` registers `--seed` with `default=-1` and that value keeps the RNG unseeded. Two runs of
the same configuration then differ by roughly one point of frame-name accuracy, which is enough to
manufacture or erase a methodological difference. Every run behind a reported number used `--seed 42`
(second seeds where an arm needed more than one draw), and the trainer prints the seed it actually used
(`train_task3.py:362`).

## 7. Three more things that bit us

- **Path convention.** The scripts name `/work/workspace/ccl25/baseline*/...` literally, and directory
  names are not normalised in this repository on purpose — renaming `baseline/` would break every
  chain script, so the folder names are load-bearing. The `${CCL25_HOST}` placeholder in the chain
  drivers is the machine the container runs on, not a path component.
- **GPU jobs must be started from a per-job interpreter name.** `nvidia-smi`'s process column shows the
  executable path, so launching with bare `python3` declares nothing about who owns the card and makes
  another team's job indistinguishable from ours. The ladder builds hard-linked names
  (`ccl25_dec1_<res>` etc.) and prints them before starting
  ([runbash/decode_ladder_res_1106.sh:60-66](../runbash/decode_ladder_res_1106.sh)).
- **Never edit a chain script while bash is still reading it.** Bash continues a script by byte offset;
  rewriting the file mid-run makes it execute from the wrong offset. Copy, edit the copy, or wait for
  the run to finish.
- **`docker exec -d` returning success is not a live process.** Within two minutes check the process
  list, that the log is growing, and that memory is allocated on the card. A detached start that
  reported no error can still be a job that died in the first seconds, and the chain log then stops
  without ever printing its completion line.

## 8. The upstream baseline's own `requirements.txt` is wrong

`baseline/requirements.txt` is redistributed exactly as the competition distributed it, for
provenance. It pins an older torch/transformers pair (`torch==1.13.0`, `transformers==4.24.0`) than the
one every reported run used, and lists `codecs` and `json`, which are standard-library modules, so
`pip install -r baseline/requirements.txt` fails. Use the repository root's `requirements.txt`.
