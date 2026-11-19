# CS781 Assignments 1-6

Anirudh Bankhede | IIT Kanpur | Roll 240131

Coursework notebooks, implementations, experiments, evaluation records and submission packages for six CS781 assignments.

| Assignment | Contents |
|---|---|
| 01 | [Language modeling, Word2Vec, NNLM and BPE](assignment-01/GITHUB_GUIDE.md) |
| 02 | [Recurrent networks, translation and attention](assignment-02/GITHUB_GUIDE.md) |
| 03 | [Decoder transformers, attention variants, RoPE and optimizers](assignment-03/GITHUB_GUIDE.md) |
| 04 | [Decoding, prompting, PEFT, quantization and distillation](assignment-04/GITHUB_GUIDE.md) |
| 05 | [Interpretability, grokking and generalization](assignment-05/GITHUB_GUIDE.md) |
| 06 | [Retrieval augmented generation and agentic AI](assignment-06/GITHUB_GUIDE.md) |

## Files and downloads

Source files and saved results are browsable in the assignment folders. Large files, checkpoints and ZIP packages are attached to the [coursework release](https://github.com/anirudh3810/cs781-assignments/releases/tag/coursework-2026-11-19). [ARTIFACTS.md](ARTIFACTS.md) lists the original paths and direct downloads. `publication-manifest.json` records file sizes and SHA-256 checksums.

To restore every released file to its original assignment path:

```bash
python tools/download_artifacts.py
```

Use each assignment's README or notebook setup cells for dependencies; the assignments use different environments. The archive includes previous and backup versions already present locally. Credentials are redacted in publication copies. Virtual environments, bytecode and regenerable Assignment 6 downloaded-model/cache files are excluded; see [PUBLICATION_NOTES.md](PUBLICATION_NOTES.md).

The associated seven-phase project is [CS781 Capstone: Legal Intelligence](https://github.com/anirudh3810/cs781-capstone-legal-intelligence).
