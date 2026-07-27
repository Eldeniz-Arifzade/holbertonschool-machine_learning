#!/usr/bin/env python3
"""Trains a Gensim Word2Vec model."""

from gensim.models import Word2Vec


def word2vec_model(sentences, vector_size=100, min_count=5, window=5,
                   negative=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create, build, and train a Word2Vec model.

    Args:
        sentences: Tokenized sentences on which to train.
        vector_size: Dimensionality of the word vectors.
        min_count: Minimum word frequency included in training.
        window: Maximum distance between target and context words.
        negative: Number of negative samples.
        cbow: Use CBOW when true and Skip-gram when false.
        epochs: Number of training iterations over the corpus.
        seed: Seed for the random number generator.
        workers: Number of worker threads.

    Returns:
        The trained Gensim Word2Vec model.
    """
    model = Word2Vec(
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
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
