#!/usr/bin/env python3
"""Yolo v3 object detection module."""

import tensorflow.keras as K


class Yolo:
    """Yolo class uses the Yolo v3 algorithm for object detection."""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Initialize the Yolo object.

        Args:
            model_path (str): Path to the Darknet Keras model.
            classes_path (str): Path to file containing class names.
            class_t (float): Box score threshold for filtering.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes for predictions.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
