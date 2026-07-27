#!/usr/bin/env python3
"""Converts Gensim word vectors to a Keras embedding layer."""

from tensorflow.keras.layers import Embedding


def gensim_to_keras(model):
    """Convert a trained Gensim Word2Vec model to a Keras layer.

    Args:
        model: Trained Gensim Word2Vec model.

    Returns:
        A trainable Keras Embedding layer initialized with the vectors.
    """
    vectors = model.wv.vectors

    return Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        weights=[vectors],
        trainable=True,
    )
