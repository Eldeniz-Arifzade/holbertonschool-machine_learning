#!/usr/bin/env python3
""" Convolutional backward propagation """
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs backward propagation over a convolutional layer

    Arguments:
    - dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) with gradients of the output
    - A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    - W: numpy.ndarray of shape (kh, kw, c_prev, c_new)
    - b: numpy.ndarray of shape (1, 1, 1, c_new)
    - padding: "same" or "valid"
    - stride: tuple (sh, sw)

    Returns:
    - dA_prev: gradients wrt previous layer
    - dW: gradients wrt kernels
    - db: gradients wrt biases
    """

    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, c_new = W.shape
    _, h_new, w_new, _ = dZ.shape
    sh, sw = stride

    # ---- Padding calculation (matches typical forward pass) ----
    if padding == "same":
        ph = max((h_prev - 1) * sh + kh - h_prev, 0)
        pw = max((w_prev - 1) * sw + kw - w_prev, 0)
        pad_top = ph // 2
        pad_bottom = ph - pad_top
        pad_left = pw // 2
        pad_right = pw - pad_left
    else:
        pad_top = pad_bottom = pad_left = pad_right = 0

    # ---- Pad inputs ----
    A_prev_padded = np.pad(
        A_prev,
        ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant'
    )

    dA_prev_padded = np.zeros_like(A_prev_padded)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)  # vectorized db

    # ---- Main loops to compute dA_prev and dW ----
    for e in range(m):  # each example
        for i in range(h_new):
            for j in range(w_new):
                for c in range(c_new):
                    vert_start = i * sh
                    vert_end = vert_start + kh
                    horiz_start = j * sw
                    horiz_end = horiz_start + kw

                    # Slice of the padded input
                    A_slice = A_prev_padded[e,
                                            vert_start:vert_end,
                                            horiz_start:horiz_end,
                                            :]

                    # Compute gradients
                    dW[:, :, :, c] += A_slice * dZ[e, i, j, c]
                    dA_prev_padded[e,
                                   vert_start:vert_end,
                                   horiz_start:horiz_end,
                                   :] += W[:, :, :, c] * dZ[e, i, j, c]

    # ---- Unpad dA_prev if needed ----
    if padding == "same":
        dA_prev = dA_prev_padded[:,
                                 pad_top:dA_prev_padded.shape[1]-pad_bottom,
                                 pad_left:dA_prev_padded.shape[2]-pad_right,
                                 :]
    else:
        dA_prev = dA_prev_padded

    return dA_prev, dW, db