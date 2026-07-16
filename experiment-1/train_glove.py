"""Train a small GloVe-style model on the CS224n Assignment 1 IMDB corpus."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from sklearn.decomposition import TruncatedSVD


START_TOKEN = "<START>"
END_TOKEN = "<END>"
DEFAULT_WORDS = [
    "movie",
    "book",
    "mysterious",
    "story",
    "fascinating",
    "good",
    "interesting",
    "large",
    "massive",
    "huge",
]


def read_corpus(num_samples: int) -> list[list[str]]:
    """Load and preprocess reviews exactly like the assignment notebook."""
    imdb_dataset = load_dataset("stanfordnlp/imdb", name="plain_text")
    files = imdb_dataset["train"]["text"][:num_samples]
    return [
        [START_TOKEN]
        + [re.sub(r"[^\w]", "", word.lower()) for word in review.split(" ")]
        + [END_TOKEN]
        for review in files
    ]


def distinct_words(corpus: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    words = sorted({word for review in corpus for word in review})
    return words, {word: index for index, word in enumerate(words)}


def build_cooccurrences(
    corpus: list[list[str]], word2ind: dict[str, int], window_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts: Counter[tuple[int, int]] = Counter()

    for review in corpus:
        token_ids = [word2ind[word] for word in review]
        for center_pos, center_id in enumerate(token_ids):
            start = max(0, center_pos - window_size)
            end = min(len(token_ids), center_pos + window_size + 1)
            for context_pos in range(start, end):
                if context_pos == center_pos:
                    continue
                counts[(center_id, token_ids[context_pos])] += 1.0

    cooccurrences = np.array(
        [(center_id, context_id, count) for (center_id, context_id), count in counts.items()],
        dtype=np.float64,
    )
    return (
        cooccurrences[:, 0].astype(np.int64),
        cooccurrences[:, 1].astype(np.int64),
        cooccurrences[:, 2],
    )


def train_glove(
    center_ids: np.ndarray,
    context_ids: np.ndarray,
    counts: np.ndarray,
    vocab_size: int,
    embedding_dim: int,
    epochs: int,
    learning_rate: float,
    x_max: float,
    alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]]]:
    rng = np.random.default_rng(seed)
    scale = 0.5 / embedding_dim
    word_vectors = rng.uniform(-scale, scale, (vocab_size, embedding_dim))
    context_vectors = rng.uniform(-scale, scale, (vocab_size, embedding_dim))
    word_biases = np.zeros(vocab_size)
    context_biases = np.zeros(vocab_size)

    grad_sq_word = np.ones_like(word_vectors)
    grad_sq_context = np.ones_like(context_vectors)
    grad_sq_word_bias = np.ones_like(word_biases)
    grad_sq_context_bias = np.ones_like(context_biases)

    weights = np.minimum((counts / x_max) ** alpha, 1.0)
    log_counts = np.log(counts)
    indices = np.arange(counts.shape[0])
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        start_time = time.perf_counter()

        for idx in indices:
            center_id = center_ids[idx]
            context_id = context_ids[idx]
            weight = weights[idx]

            prediction = (
                np.dot(word_vectors[center_id], context_vectors[context_id])
                + word_biases[center_id]
                + context_biases[context_id]
            )
            diff = prediction - log_counts[idx]
            weighted_diff = weight * diff
            epoch_loss += 0.5 * weight * diff * diff

            word_grad = weighted_diff * context_vectors[context_id]
            context_grad = weighted_diff * word_vectors[center_id]

            word_vectors[center_id] -= learning_rate * word_grad / np.sqrt(
                grad_sq_word[center_id]
            )
            context_vectors[context_id] -= learning_rate * context_grad / np.sqrt(
                grad_sq_context[context_id]
            )
            word_biases[center_id] -= learning_rate * weighted_diff / math.sqrt(
                grad_sq_word_bias[center_id]
            )
            context_biases[context_id] -= learning_rate * weighted_diff / math.sqrt(
                grad_sq_context_bias[context_id]
            )

            grad_sq_word[center_id] += word_grad * word_grad
            grad_sq_context[context_id] += context_grad * context_grad
            grad_sq_word_bias[center_id] += weighted_diff * weighted_diff
            grad_sq_context_bias[context_id] += weighted_diff * weighted_diff

        average_loss = epoch_loss / counts.shape[0]
        elapsed = time.perf_counter() - start_time
        history.append(
            {"epoch": epoch, "average_loss": average_loss, "seconds": elapsed}
        )
        print(
            f"epoch {epoch:03d}/{epochs}: loss={average_loss:.6f} time={elapsed:.2f}s",
            flush=True,
        )

    return word_vectors, context_vectors, word_biases, context_biases, history


def cosine_neighbors(
    vectors: np.ndarray,
    words: list[str],
    word2ind: dict[str, int],
    query_words: list[str],
    top_k: int,
) -> dict[str, list[tuple[str, float]]]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    neighbors: dict[str, list[tuple[str, float]]] = {}

    for query in query_words:
        if query not in word2ind:
            continue
        query_index = word2ind[query]
        similarities = normalized @ normalized[query_index]
        best_indices = np.argsort(-similarities)
        best = [
            (words[index], float(similarities[index]))
            for index in best_indices
            if index != query_index
        ][:top_k]
        neighbors[query] = best

    return neighbors


def save_plot(
    vectors: np.ndarray,
    words: list[str],
    word2ind: dict[str, int],
    query_words: list[str],
    output_path: Path,
) -> None:
    available_words = [word for word in query_words if word in word2ind]
    if len(available_words) < 2:
        return

    reduced = TruncatedSVD(n_components=2, n_iter=10, random_state=0).fit_transform(
        vectors
    )
    reduced_lengths = np.linalg.norm(reduced, axis=1)
    normalized = reduced / np.maximum(reduced_lengths[:, np.newaxis], 1e-12)

    plt.figure(figsize=(10, 5))
    for word in available_words:
        x_coord, y_coord = normalized[word2ind[word]]
        plt.scatter(x_coord, y_coord, marker="x", color="red")
        plt.text(x_coord, y_coord, word, fontsize=9)
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train small GloVe-style embeddings on the assignment IMDB corpus."
    )
    parser.add_argument("--num-samples", type=int, default=150)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--x-max", type=float, default=100.0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("experiment-1/outputs"))
    parser.add_argument("--plot-words", nargs="+", default=DEFAULT_WORDS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.perf_counter()
    print("Loading IMDB corpus...", flush=True)
    corpus = read_corpus(args.num_samples)
    token_count = sum(len(review) for review in corpus)
    words, word2ind = distinct_words(corpus)
    print(f"reviews={len(corpus)} tokens={token_count} vocab={len(words)}", flush=True)

    print("Building sparse co-occurrence counts...", flush=True)
    center_ids, context_ids, counts = build_cooccurrences(
        corpus, word2ind, args.window_size
    )
    print(f"nonzero co-occurrences={counts.shape[0]}", flush=True)

    print("Training GloVe-style embeddings...", flush=True)
    train_start = time.perf_counter()
    word_vectors, context_vectors, word_biases, context_biases, history = train_glove(
        center_ids=center_ids,
        context_ids=context_ids,
        counts=counts,
        vocab_size=len(words),
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        x_max=args.x_max,
        alpha=args.alpha,
        seed=args.seed,
    )
    train_seconds = time.perf_counter() - train_start
    embeddings = word_vectors + context_vectors

    neighbors = cosine_neighbors(
        embeddings, words, word2ind, args.plot_words, top_k=args.top_k
    )
    total_seconds = time.perf_counter() - total_start

    np.savez_compressed(
        args.output_dir / "small_glove_embeddings.npz",
        embeddings=embeddings,
        word_vectors=word_vectors,
        context_vectors=context_vectors,
        word_biases=word_biases,
        context_biases=context_biases,
    )
    (args.output_dir / "vocab.json").write_text(json.dumps(words, indent=2) + "\n")

    metrics = {
        "num_samples": args.num_samples,
        "window_size": args.window_size,
        "embedding_dim": args.embedding_dim,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "x_max": args.x_max,
        "alpha": args.alpha,
        "seed": args.seed,
        "reviews": len(corpus),
        "tokens": token_count,
        "vocab_size": len(words),
        "nonzero_cooccurrences": int(counts.shape[0]),
        "training_seconds": train_seconds,
        "total_seconds": total_seconds,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    neighbor_lines = []
    for query, results in neighbors.items():
        neighbor_lines.append(f"{query}:")
        for word, score in results:
            neighbor_lines.append(f"  {word}\t{score:.4f}")
        neighbor_lines.append("")
    (args.output_dir / "nearest_neighbors.txt").write_text(
        "\n".join(neighbor_lines).rstrip() + "\n"
    )

    save_plot(
        embeddings,
        words,
        word2ind,
        args.plot_words,
        args.output_dir / "assignment_words_svd.png",
    )

    print(f"Saved outputs to {args.output_dir}", flush=True)
    print(f"training_seconds={train_seconds:.2f}", flush=True)
    print(f"total_seconds={total_seconds:.2f}", flush=True)


if __name__ == "__main__":
    main()
