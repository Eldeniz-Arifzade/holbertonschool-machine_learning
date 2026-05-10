#!/usr/bin/env python3
"""Neural Style Transfer module."""

import tensorflow as tf
import numpy as np


class NST:
    """Performs tasks for neural style transfer."""

    style_layers = ['block1_conv1', 'block2_conv1',
                    'block3_conv1', 'block4_conv1',
                    'block5_conv1']

    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image,
                 alpha=1e4, beta=1):
        """
        Initialize NST object.

        Args:
            style_image (np.ndarray): Style reference image.
            content_image (np.ndarray): Content reference image.
            alpha (float): Content cost weight.
            beta (float): Style cost weight.
        """
        if (not isinstance(style_image, np.ndarray) or
                len(style_image.shape) != 3 or
                style_image.shape[2] != 3):
            raise TypeError(
                "style_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(content_image, np.ndarray) or
                len(content_image.shape) != 3 or
                content_image.shape[2] != 3):
            raise TypeError(
                "content_image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        if (not isinstance(alpha, (int, float)) or
                alpha < 0):
            raise TypeError(
                "alpha must be a non-negative number"
            )

        if (not isinstance(beta, (int, float)) or
                beta < 0):
            raise TypeError(
                "beta must be a non-negative number"
            )

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)

        self.alpha = alpha
        self.beta = beta

        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescale image for neural style transfer.

        Args:
            image (np.ndarray): Image of shape (h, w, 3).

        Returns:
            tf.Tensor: Scaled image tensor.
        """
        if (not isinstance(image, np.ndarray) or
                len(image.shape) != 3 or
                image.shape[2] != 3):
            raise TypeError(
                "image must be a numpy.ndarray "
                "with shape (h, w, 3)"
            )

        h, w, _ = image.shape

        max_dim = 512

        if h > w:
            new_h = max_dim
            new_w = int((w / h) * max_dim)
        else:
            new_w = max_dim
            new_h = int((h / w) * max_dim)

        resized = tf.image.resize(
            image,
            (new_h, new_w),
            method=tf.image.ResizeMethod.BICUBIC
        )

        resized = resized / 255.0

        resized = tf.clip_by_value(resized, 0.0, 1.0)

        resized = tf.expand_dims(resized, axis=0)

        return resized

    def load_model(self):
        """
        Create the VGG19 model for neural style transfer.

        The model outputs the activations from the style
        layers followed by the content layer.
        """
        vgg = tf.keras.applications.VGG19(
            include_top=False,
            weights='imagenet'
        )

        vgg.trainable = False

        outputs = [vgg.get_layer(name).output
                   for name in self.style_layers]

        outputs.append(
            vgg.get_layer(self.content_layer).output
        )

        model = tf.keras.models.Model(
            inputs=vgg.input,
            outputs=outputs
        )

        for layer in model.layers:
            if isinstance(layer, tf.keras.layers.MaxPooling2D):
                config = layer.get_config()

                avg_pool = tf.keras.layers.AveragePooling2D(
                    pool_size=config['pool_size'],
                    strides=config['strides'],
                    padding=config['padding'],
                    name=layer.name
                )

                x = avg_pool(layer.input)

                for node in layer._outbound_nodes:
                    node.outbound_layer._inbound_nodes = []

        self.model = model
        self.model.trainable = False
