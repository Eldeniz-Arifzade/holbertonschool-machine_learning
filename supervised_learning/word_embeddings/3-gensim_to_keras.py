#!/usr/bin/env python3
"""Converts Gensim word vectors to a Keras embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Gensim Word2Vec model to a Keras Embedding layer."""
    vectors = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        weights=[vectors],
        trainable=True
    )

    return embedding
