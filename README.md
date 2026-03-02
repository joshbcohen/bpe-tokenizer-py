# BPE Tokenizer

Implementation of a byte-pair encoding (BPE) tokenizer in Python. This is based off of [assignment 1](https://github.com/stanford-cs336/assignment1-basics/tree/main) from Stanford's [CS 336- Language Modeling from Scratch](https://cs336.stanford.edu/spring2025/) course. Some code and tests are pulled from the assignment's starter repository. A fork of the full assignment can be found [here](https://github.com/joshbcohen/assignment1-basics).

There is a companion repository to this in [bpe-tokenizer-pyo3](https://github.com/joshbcohen/bpe-tokenizer-pyo3), which contains a similar tokenizer implemented with Python + Rust extensions using PyO3.

## Setup

### Environment
Install `uv` [here](https://github.com/astral-sh/uv) or run `pip install uv`/`brew install uv`
Dependencies can be installed with `uv sync`.

### Run unit tests
```sh
uv run pytest
```

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```
