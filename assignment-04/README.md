# CS781 Assignment 4 — Student package

Unzip this folder to `/content/assignment_4` (Colab) or any local directory and open
`CS781_Assignment_4_Questions_updated.ipynb` from there. The notebook `cd`s into this folder automatically.

| File | Purpose | Edit? |
|---|---|---|
| `common.py` | Shared infrastructure: IMDb data contract, metrics, resumable training harness, timing/memory helpers, `save_submission`, Q3 chat generator + call cache, A-3 restore helpers | **No** |
| `download_data.py` | Downloads `Assignment4/train.parquet` and `test.parquet` from `Exploration-Lab/CS781` into `data/imdb/` | No |
| `common_grading.py`, `autograder_q1.py` … `autograder_q6.py` | Local autograders (same checks the TAs run) | No |
| `make_submission.sh` / `make_submission.ps1` | Build `CS781-A4-Name-RollNo.zip` with the required layout | No |
| `requirements.txt` | Packages used by the notebook and graders | No |
| `q1_decoding.py` … `q6_distill.py` | **Created by the notebook** (`%%writefile` cells). Your implementations live here | Yes (via the notebook) |

## Before you start

1. **Hugging Face access.** Request or accept dataset access at
   <https://huggingface.co/datasets/Exploration-Lab/CS781> with your Hugging Face account.
   Create a read token from that same account. In Kaggle, add it under **Add-ons → Secrets**
   with the label `HF_TOKEN` and enable access for this notebook; in Colab, use its Secrets
   panel. Never paste a token into the notebook.
2. **Q1 needs your A-3 Q5 artifacts.** Keep the A-3 ZIP beside the A-4 package,
   or add an extracted `CS781-A3-*` folder with `submissions/q5/config.json`
   and `model.pt`. The Q1 settings cell detects either layout.
3. Use a GPU runtime. On Kaggle T4 x2, Q1 decoding and Q3 prompting distribute
   independent inference jobs across both GPUs. Q2, Q4 and Q6 use data-parallel
   classifier/student training. Q5 uses synchronized two-replica QLoRA training,
   with one example per GPU in each two-example microbatch. Validation and a few
   single-call checks use one GPU. T4 training uses FP16 autocast; BF16 is
   reserved for GPUs with native BF16 support. Q5 and Q6 require CUDA
   (bitsandbytes 4-bit). Actual utilization depends on prompt length, caching,
   data loading and the active stage; two GPUs do not guarantee a 2x speedup.
   `requirements.txt` pins the library versions recorded in the successful
   Kaggle setup run, and the first notebook cell applies them to an existing
   Kaggle working copy before installation. The setup cell removes Kaggle's
   incompatible preinstalled `torchao` only when its version is below the
   minimum accepted by PEFT, then runs a small LoRA dependency check.
4. Fill `STUDENT_NAME_INPUT`, `ROLL_NUMBER_INPUT`, and `PERSISTENT_ROOT_INPUT` in the
   notebook configuration cell (environment variables with matching `CS781_` names
   can override them). On Kaggle, the configured `/kaggle/working/cs781_a4` path
   requires **Session Persistence: Files only** or **Variables & Files** in the
   notebook settings before running. Select `PROFILE = "core"` or `"low_resource"`, then Run All.
   To continue a previous Kaggle run, keep both `/kaggle/working/CS781_ASSIGNMENT4`
   and `/kaggle/working/cs781_a4`, and leave `PROFILE` and `PERSISTENT_ROOT_INPUT`
   at their previous values. Completed stages are checked against their manifests;
   interrupted training restores its latest compatible `last.pt` checkpoint.
   After the chat-token counting fix, Q3.5's earlier context grid is regenerated
   once because its recorded prompt lengths were incorrect; its old call records
   are archived under `q3_5_context_records_before_token_fix.json`. Other
   compatible completed stages and training checkpoints are reused.
   If Q6 has already reached its final CE checkpoint, Run All restores that
   checkpoint and finishes the result export without repeating its training.
   Undefined AMP gradient diagnostics are recorded as `null`; nonfinite model
   predictions or evaluation metrics still stop the run.
   Kaggle may keep live notebook outputs in its editor while the package's
   `.ipynb` file remains the uploaded source. The final cell first searches for
   a saved executed copy from the current Run All, then runs all six graders.
   It records every grader result before reporting any failure.
   If its result is `awaiting_executed_notebook`, save/export the executed
   notebook with outputs, attach that file with Kaggle **Add Input**, and rerun
   only the final cell. It finds the matching `.ipynb` under `/kaggle/input`,
   copies it to the package directory as `notebook.ipynb`, then grades and
   builds the ZIP. You can also put it directly in the package directory.
   The experiment checkpoints are retained.

## Required submission layout

```text
CS781-A4-Firstname-Lastname-RollNo.zip
    notebook.ipynb
    common.py  requirements.txt  download_data.py
    q1_decoding.py  q2_classifier.py  q3_prompting.py  q4_peft.py  q5_qlora.py  q6_distill.py
    submissions/
        q1/  metrics.json  q1_decoding.py  common.py
        q2/  metrics.json  weights.pt  q2_classifier.py  common.py
        q3/  metrics.json  q3_prompting.py  common.py
        q4/  metrics.json  weights.pt  q4_peft.py  common.py
        q5/  metrics.json  weights.pt  q5_qlora.py  common.py
        q6/  metrics.json  weights.pt  q6_distill.py  common.py
```

`common.save_submission(...)` (called at the end of every question) writes each `submissions/qN/`
folder, including a snapshot of the module that produced the weights.

## Run the autograders yourself

```bash
python download_data.py                          # once, if you want the data cached locally
for q in 1 2 3 4 5 6; do python autograder_q$q.py --student_dir . ; done
```

Options: `--skip_reload` (Q2/Q4/Q5/Q6: skip downloading base models and re-predicting),
`--skip_tokenizer` (Q5: skip the Qwen tokenizer check). Q5's reload check runs only on a CUDA machine.
Passing the autograders does not replace the written analysis — they catch broken or missing
implementations, not the quality of your conclusions.
