# Experiment 1: Small GloVe Training

This experiment trains GloVe-style word embeddings from scratch on the same small
IMDB movie review corpus used for the Assignment 1 co-occurrence matrix.

The default setup mirrors `assignment-1/student/exploring_word_vectors.ipynb`:

- Dataset: `stanfordnlp/imdb`, `plain_text`
- Split: first 150 training reviews
- Preprocessing: lowercase, strip non-word characters with `re.sub(r'[^\w]', '', word)`
- Tokens: add `<START>` and `<END>` around every review
- Context window: 4 words on each side

## Setup

Use the Assignment 1 environment if available:

```bash
conda activate cs224n
```

That environment is defined in `assignment-1/student/env.yml` and already includes
the required packages: `datasets`, `numpy`, `matplotlib`, and `scikit-learn`.

## Run

From the repo root:

```bash
python experiment-1/train_glove.py
```

On the default 150-review experiment, the observed runtime on this machine was
about 80 seconds total, with about 79 seconds spent in training.

Useful overrides:

```bash
python experiment-1/train_glove.py --epochs 25 --embedding-dim 50
python experiment-1/train_glove.py --num-samples 150 --window-size 4 --seed 0
```

## Outputs

Generated files are written to `experiment-1/outputs`:

- `small_glove_embeddings.npz`: learned embeddings and model parameters
- `vocab.json`: sorted vocabulary
- `metrics.json`: hyperparameters, corpus statistics, timing, and loss history
- `nearest_neighbors.txt`: cosine nearest neighbors for selected assignment words
- `assignment_words_svd.png`: 2D Truncated SVD plot for selected assignment words

## Notes

This is an intentionally small, readable implementation for experimentation. It is
not meant to match the scale or quality of pretrained GloVe vectors, which are
trained on billions of tokens. With only 150 reviews, training should be quick, but
the learned embeddings will be noisy and highly specific to this small movie-review
sample.
