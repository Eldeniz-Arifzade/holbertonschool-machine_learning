#!/usr/bin/env python3
"""Yolo v3 object detection module."""

import numpy as np
import tensorflow.keras as K


class Yolo:
    """Yolo class uses the Yolo v3 algorithm for object detection."""

    def __init__(self, model_path, classes_path,
                 class_t, nms_t, anchors):
        """
        Initialize the Yolo object.

        Args:
            model_path (str): Path to Darknet Keras model.
            classes_path (str): Path to class names file.
            class_t (float): Box score threshold.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Anchor boxes.
        """
        self.model = K.models.load_model(model_path)

        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]

        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    @staticmethod
    def sigmoid(x):
        """
        Calculate sigmoid activation.

        Args:
            x (numpy.ndarray): Input array.

        Returns:
            numpy.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet outputs.

        Args:
            outputs (list): Predictions from Darknet model.
            image_size (numpy.ndarray): Original image size
                                        [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        image_h = image_size[0]
        image_w = image_size[1]

        for i, output in enumerate(outputs):
            grid_h = output.shape[0]
            grid_w = output.shape[1]
            anchor_boxes = output.shape[2]

            t_x = output[..., 0]
            t_y = output[..., 1]
            t_w = output[..., 2]
            t_h = output[..., 3]

            c_x = np.arange(grid_w)
            c_y = np.arange(grid_h)

            c_x, c_y = np.meshgrid(c_x, c_y)

            c_x = np.repeat(c_x[..., np.newaxis], anchor_boxes, axis=-1)
            c_y = np.repeat(c_y[..., np.newaxis], anchor_boxes, axis=-1)

            b_x = (self.sigmoid(t_x) + c_x) / grid_w
            b_y = (self.sigmoid(t_y) + c_y) / grid_h

            anchor_w = self.anchors[i, :, 0]
            anchor_h = self.anchors[i, :, 1]

            b_w = (anchor_w * np.exp(t_w)) / input_w
            b_h = (anchor_h * np.exp(t_h)) / input_h

            x1 = (b_x - (b_w / 2)) * image_w
            y1 = (b_y - (b_h / 2)) * image_h
            x2 = (b_x + (b_w / 2)) * image_w
            y2 = (b_y + (b_h / 2)) * image_h

            box = np.zeros(output[..., :4].shape)
            box[..., 0] = x1
            box[..., 1] = y1
            box[..., 2] = x2
            box[..., 3] = y2

            boxes.append(box)

            box_confidence = self.sigmoid(output[..., 4:5])
            box_confidences.append(box_confidence)

            box_class_prob = self.sigmoid(output[..., 5:])
            box_class_probs.append(box_class_prob)

        return (boxes, box_confidences, box_class_probs)
