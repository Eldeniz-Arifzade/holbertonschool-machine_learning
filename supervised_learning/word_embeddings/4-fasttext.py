#!/usr/bin/env python3
"""Trains a Gensim FastText model."""

import gensim


def fasttext_model(sentences, vector_size=100, min_count=5, negative=5,
                   window=5, cbow=True, epochs=5, seed=0, workers=1):
    """Create and train a FastText model."""
    return gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        negative=negative,
        window=window,
        sg=not cbow,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
