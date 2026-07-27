#!/usr/bin/env python3
"""Trains a Gensim FastText model."""

from gensim.models import FastText


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a FastText model.

    Args:
        sentences: Tokenized sentences on which to train.
        vector_size: Dimensionality of the word vectors.
        min_count: Minimum word frequency included in training.
        negative: Number of negative samples.
        window: Maximum distance between target and context words.
        cbow: Use CBOW when true and Skip-gram when false.
        epochs: Number of training iterations over the corpus.
        seed: Seed for the random number generator.
        workers: Number of worker threads.

    Returns:
        The trained Gensim FastText model.
    """
    model = FastText(
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=not cbow,
        seed=seed,
        workers=workers,
    )
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=epochs,
    )

    return model
