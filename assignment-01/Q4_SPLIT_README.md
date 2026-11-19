# Q4 Parallel Workflow

This Q4 split is now aligned with the actual number of trainable models in Question 4:
one English NNLM and one Hindi NNLM.

Files to run in parallel:
- `Q4_Training_EN.ipynb`
- `Q4_Training_HI.ipynb`

How the split works:
1. `Q4_Training_EN.ipynb`
   - trains only the English NNLM
   - saves a resumable checkpoint every 5 epochs to `checkpoint_nnlm_en.pt`
   - resumes automatically if that checkpoint is already present
   - computes all English-side Q4 outputs
   - creates `q4_en_artifacts.zip` for download

2. `Q4_Training_HI.ipynb`
   - trains only the Hindi NNLM
   - saves a resumable checkpoint every 5 epochs to `checkpoint_nnlm_hi.pt`
   - resumes automatically if that checkpoint is already present
   - computes all Hindi-side Q4 outputs
   - creates `q4_hi_artifacts.zip` for download

Kaggle notes:
- Open the English notebook in one Kaggle session and the Hindi notebook in another Kaggle session.
- Set accelerator to GPU in both sessions.
- The English notebook is pinned to `DEVICE_ID = 0`.
- The Hindi notebook is pinned to `DEVICE_ID = 1`, but if that session only exposes one GPU it will still run on the visible GPU.
- You only need to upload the notebook itself. It recreates `q4_parallel_runtime.py` automatically inside the session before importing it.
- Files added through Kaggle's Input pane are copied automatically into the working directory before the run starts.

Files produced per language notebook:
- `nnlm_<lang>.pt`
- `checkpoint_nnlm_<lang>.pt`
- `q4_train_state_<lang>.pkl`
- `q4_train_summary_<lang>.json`
- `q4_convergence_<lang>.png`
- `q4_perplexity_table_<lang>.csv`
- `q4_parameter_table_<lang>.csv`
- `q4_sparsity_table_<lang>.csv`
- `q4_mixture_grid_<lang>.csv`
- `q4_metrics_summary_<lang>.json`
- `analogy_results_nnlm_<lang>.csv`
- `q4_embedding_pca_<lang>.png`
- `q4_word2vec_compare_<lang>.csv`
- `q4_word2vec_proxy_<lang>.csv`
- `q4_embedding_compare_<lang>.png`
- `q4_analysis_summary_<lang>.json`
- `q4_<lang>_artifacts.zip`

Final combination:
- After both notebooks finish, combine the English and Hindi outputs later for the final submission tables or write-up if needed.
- This is the correct split because Q4 has only two actual training runs.
