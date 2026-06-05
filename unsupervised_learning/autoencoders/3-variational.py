#!/usr/bin/env python3
"""Module for creating a variational autoencoder."""
import tensorflow.keras as keras
import tensorflow as tf


class Sampling(keras.layers.Layer):
    """Reparameterization sampling layer for the VAE latent space."""

    def call(self, inputs):
        """Sample z using the reparameterization trick.

        Args:
            inputs (tuple): (z_mean, z_log_var) tensors.

        Returns:
            tensor: Sampled latent vector z = mean + eps * exp(log_var / 2).
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(z_log_var / 2) * epsilon


class KLLossLayer(keras.layers.Layer):
    """Layer that computes and registers KL divergence loss, passing z through."""

    def call(self, inputs):
        """Compute KL loss and add it; return z unchanged.

        Args:
            inputs (list): [z, z_mean, z_log_var] tensors.

        Returns:
            tensor: z unchanged.
        """
        z, z_mean, z_log_var = inputs
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        return z


def autoencoder(input_dims, hidden_layers, latent_dims):
    """Create a variational autoencoder.

    Args:
        input_dims (int): Dimensions of the model input.
        hidden_layers (list): Number of nodes for each hidden layer in the
            encoder. The hidden layers are reversed for the decoder.
        latent_dims (int): Dimensions of the latent space representation.

    Returns:
        tuple: (encoder, decoder, auto)
            - encoder: outputs (z, z_mean, z_log_var)
            - decoder: the decoder model
            - auto: the full autoencoder model compiled with adam and
              binary cross-entropy + KL divergence loss
    """
    # --- Encoder ---
    encoder_input = keras.Input(shape=(input_dims,))
    x = encoder_input
    for nodes in hidden_layers:
        x = keras.layers.Dense(nodes, activation='relu')(x)
    z_mean = keras.layers.Dense(latent_dims, activation=None)(x)
    z_log_var = keras.layers.Dense(latent_dims, activation=None)(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_input, [z, z_mean, z_log_var])

    # --- Decoder ---
    decoder_input = keras.Input(shape=(latent_dims,))
    x = decoder_input
    for nodes in reversed(hidden_layers):
        x = keras.layers.Dense(nodes, activation='relu')(x)
    decoder_output = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(decoder_input, decoder_output)

    # --- Autoencoder: inline graph so KLLossLayer is a direct node ---
    auto_input = keras.Input(shape=(input_dims,))
    # Replay encoder layers directly (not as sub-model) so losses propagate
    h = auto_input
    for nodes in hidden_layers:
        h = keras.layers.Dense(nodes, activation='relu')(h)
    z_mean_auto = keras.layers.Dense(latent_dims, activation=None)(h)
    z_log_var_auto = keras.layers.Dense(latent_dims, activation=None)(h)
    z_auto = Sampling()([z_mean_auto, z_log_var_auto])
    # KL loss registered via layer node in the graph
    z_auto = KLLossLayer()([z_auto, z_mean_auto, z_log_var_auto])
    # Replay decoder layers directly
    d = z_auto
    for nodes in reversed(hidden_layers):
        d = keras.layers.Dense(nodes, activation='relu')(d)
    auto_output = keras.layers.Dense(input_dims, activation='sigmoid')(d)
    auto = keras.Model(auto_input, auto_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
