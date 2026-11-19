# Q3 Split Training Workflow

Notebook assignments:
- `Q3_Training_GPU_1.ipynb`: `EN-Softmax` and `HI-NS`
- `Q3_Training_GPU_2.ipynb`: `EN-NS` and `HI-Softmax`
- `Q3_Training_GPU_3.ipynb`: `EN-HS` and `HI-HS`

Shared-corpus procedure:
1. Run any one notebook until it creates `shared_english_corpus.csv` and `shared_hindi_corpus.csv`.
2. Download those two CSVs.
3. Upload the same two CSVs into the other GPU sessions before running their shared setup cell if you want guaranteed cache reuse.
4. If you skip the upload step, the notebooks still rebuild the corpora deterministically from the same dataset and fixed seed.

Crash recovery:
1. Keep the latest `checkpoint_<label>.pt` file for the model you were training.
2. In a fresh runtime, restore the shared corpus CSVs first.
3. Run the notebook cells through the checkpoint-helper cell.
4. Run `upload_checkpoint("<label>")` for the interrupted model.
5. Re-run that model's training cell. It will resume from the saved epoch after validating `vocab_size` and `embed_dim`.

Download after each notebook finishes:
- GPU 1: `en_embeddings_softmax.pkl`, `checkpoint_EN-Softmax.pt`, `hi_embeddings_ns.pkl`, `checkpoint_HI-NS.pt`, `shared_english_corpus.csv`, `shared_hindi_corpus.csv`
- GPU 2: `en_embeddings_ns.pkl`, `checkpoint_EN-NS.pt`, `hi_embeddings_softmax.pkl`, `checkpoint_HI-Softmax.pt`, `shared_english_corpus.csv`, `shared_hindi_corpus.csv`
- GPU 3: `en_embeddings_hs.pkl`, `checkpoint_EN-HS.pt`, `hi_embeddings_hs.pkl`, `checkpoint_HI-HS.pt`, `shared_english_corpus.csv`, `shared_hindi_corpus.csv`

Six `.pkl` files needed later:
- `en_embeddings_softmax.pkl`
- `en_embeddings_ns.pkl`
- `en_embeddings_hs.pkl`
- `hi_embeddings_softmax.pkl`
- `hi_embeddings_ns.pkl`
- `hi_embeddings_hs.pkl`

Use `Q3_combine_loader.py` in the final submission notebook to load the six artifacts back into the expected variable names without retraining.
