# Q5 Split Training Workflow

This Question 5 split is organized the same way as your Q4 split:
- `Q5_Training_EN.ipynb` trains only the English subword NNLM
- `Q5_Training_HI.ipynb` trains only the Hindi subword NNLM

What each notebook does:
- trains one byte-level BPE tokenizer for that language
- tokenizes that language with the trained BPE model
- trains one subword NNLM for that language
- saves a resumable checkpoint every 5 epochs
- writes the exact Question 5 autograder files for that language
- writes extra comparison tables and draft written-answer files for that language
- packages everything into `q5_<lang>_artifacts.zip`

Autograder files created by the two notebooks together:
- `bpe_en.json`
- `bpe_hi.json`
- `bpe_nnlm_en.pt`
- `bpe_nnlm_hi.pt`
- `bpe_train_log_en.json`
- `bpe_train_log_hi.json`

Combined zip behavior:
- If both languages' three autograder files are present in the same working directory, the runtime also creates:
- `problem_5.zip`
- `question_5.zip`

Recommended uploads from previous results:
- `shared_english_corpus.csv`
- `shared_hindi_corpus.csv`
- `nnlm_en.pt`
- `nnlm_hi.pt`
- `q4_metrics_summary_en.json`
- `q4_metrics_summary_hi.json`

Fallback uploads if the JSON summaries are missing:
- `q4_perplexity_table_en.csv`
- `q4_perplexity_table_hi.csv`

Alternative bundle uploads:
- `q4_en_artifacts.zip`
- `q4_hi_artifacts.zip`
- `question_4_bundle.zip`

Not needed from previous work:
- No Question 3 embedding files are required for Question 5.

Resume-after-crash files for Q5:
- `bpe_en.json`
- `bpe_hi.json`
- `checkpoint_bpe_nnlm_en.pt`
- `checkpoint_bpe_nnlm_hi.pt`

Important assumption in this split:
- The notebooks default to `target_vocab_size = 5000` because the assignment prompt gives 5,000 as its only explicit BPE vocabulary-size example.
- If your instructor specified a different BPE vocabulary size, change `CONFIG["target_vocab_size"]` in both notebooks before starting the real runs.
