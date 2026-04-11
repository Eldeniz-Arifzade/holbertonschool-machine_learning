#!/usr/bin/env python3
"""PCA Color Augmentation as described in the AlexNet paper."""
import tensorflow as tf


def pca_color(image, alphas):
    """Perform PCA color augmentation on an image.

    This implements the color augmentation from the AlexNet paper:
    fancy PCA over the RGB pixel values of the training set.
    For each image, the principal components of the RGB channel
    covariance matrix are computed, and a multiple of the found
    principal components (scaled by their eigenvalues and random
    alphas) is added to each pixel.

    Args:
        image: a 3D tf.Tensor of shape (H, W, 3) containing the
               image to augment. Expected dtype is uint8 or float.
        alphas: a tuple of length 3 containing the amount that each
                principal color component should change (one per
                principal component direction).

    Returns:
        The augmented image as a tf.Tensor with the same shape as
        the input, clipped to [0, 255] and cast back to uint8.
    """
    # Cast to float32 and normalize to [0, 1]
    img = tf.cast(image, tf.float32) / 255.0

    # Reshape to (num_pixels, 3) — treat each pixel as a data point
    orig_shape = tf.shape(img)
    pixels = tf.reshape(img, [-1, 3])  # shape: (H*W, 3)

    # Mean-center the pixel values per channel
    mean = tf.reduce_mean(pixels, axis=0)        # shape: (3,)
    pixels_centered = pixels - mean              # shape: (H*W, 3)

    # Compute the 3x3 covariance matrix of RGB channels
    # cov = (X^T X) / (n - 1)
    n = tf.cast(tf.shape(pixels_centered)[0], tf.float32)
    cov = tf.matmul(pixels_centered, pixels_centered, transpose_a=True)
    cov = cov / (n - 1.0)                        # shape: (3, 3)

    # Eigendecomposition of the covariance matrix
    # tf.linalg.eigh returns eigenvalues in ascending order
    eigenvalues, eigenvectors = tf.linalg.eigh(cov)
    # eigenvalues: (3,), eigenvectors: (3, 3) — columns are eigenvectors

    # Build the perturbation: sum_i ( alpha_i * lambda_i * p_i )
    # where p_i is the i-th eigenvector (column), lambda_i its eigenvalue
    alphas_tensor = tf.cast(alphas, tf.float32)  # shape: (3,)

    # Scale: alpha_i * sqrt(|lambda_i|) matches AlexNet formulation
    # (eigenvalues from eigh can be near-zero/negative due to float precision)
    scales = alphas_tensor * tf.sqrt(tf.abs(eigenvalues))  # shape: (3,)

    # perturbation = eigenvectors @ scales  (matrix-vector product)
    # eigenvectors columns are the principal directions
    perturbation = tf.linalg.matvec(eigenvectors, scales)  # shape: (3,)

    # Add perturbation to every pixel
    img_augmented = img + perturbation           # broadcasts over H, W

    # Clip to valid [0, 1] range and scale back to [0, 255]
    img_augmented = tf.clip_by_value(img_augmented, 0.0, 1.0)
    img_augmented = tf.cast(img_augmented * 255.0, tf.uint8)

    return img_augmented
