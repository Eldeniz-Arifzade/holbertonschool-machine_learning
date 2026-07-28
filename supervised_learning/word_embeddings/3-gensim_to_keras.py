#!/usr/bin/env python3
"""Converts Gensim vectors to a Keras embedding layer."""

import tensorflow as tf


def gensim_to_keras(model):
    """Convert a Word2Vec model into a trainable Keras Embedding."""
    vectors = model.wv.vectors

    embedding = tf.keras.layers.Embedding(
        input_dim=vectors.shape[0],
        output_dim=vectors.shape[1],
        trainable=True
    )

    embedding.build()
    embedding.set_weights([vectors])

    return embedding
