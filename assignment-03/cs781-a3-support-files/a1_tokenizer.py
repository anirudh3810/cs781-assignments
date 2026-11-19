"""Expose your Assignment-1 BPE implementation to Assignment 3.

Reuse your A-1 tokenizer code/class here. The important change is that it must be
RETRAINED on the `train_text` supplied by Assignment 3 rather than reusing a tokenizer
trained on the Assignment-1 dataset.
"""


def train_tokenizer(train_text: str, vocab_size: int):
    """Train your A-1 BPE tokenizer on `train_text` and return the trained tokenizer.

    Required returned interface:
      tokenizer.encode(text) -> sequence of integer ids
      tokenizer.decode(ids) -> string
      tokenizer.vocab_size OR tokenizer.get_vocab_size()

    Replace this stub with your Assignment-1 tokenizer implementation/training call.
    """
    raise NotImplementedError
