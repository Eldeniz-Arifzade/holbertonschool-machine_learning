#!/usr/bin/env python3
"""Yolo v3 object detection module."""

import os
import cv2
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
        Process Darknet model outputs.

        Args:
            outputs (list): Predictions from Darknet model.
            image_size (numpy.ndarray): Original image size.

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

            c_x = np.repeat(c_x[..., np.newaxis],
                            anchor_boxes, axis=-1)
            c_y = np.repeat(c_y[..., np.newaxis],
                            anchor_boxes, axis=-1)

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

    def filter_boxes(self, boxes, box_confidences,
                     box_class_probs):
        """
        Filter boxes using objectness score and class probability.

        Args:
            boxes (list): Processed boundary boxes.
            box_confidences (list): Box confidence scores.
            box_class_probs (list): Box class probabilities.

        Returns:
            tuple: (filtered_boxes, box_classes, box_scores)
        """
        filtered_boxes = []
        box_classes = []
        box_scores = []

        for box, confidence, class_probs in zip(
                boxes, box_confidences, box_class_probs):

            scores = confidence * class_probs

            classes = np.argmax(scores, axis=-1)
            class_scores = np.max(scores, axis=-1)

            mask = class_scores >= self.class_t

            filtered_boxes.append(box[mask])
            box_classes.append(classes[mask])
            box_scores.append(class_scores[mask])

        filtered_boxes = np.concatenate(filtered_boxes, axis=0)
        box_classes = np.concatenate(box_classes, axis=0)
        box_scores = np.concatenate(box_scores, axis=0)

        return (filtered_boxes, box_classes, box_scores)

    def non_max_suppression(self, filtered_boxes,
                            box_classes, box_scores):
        """
        Apply non-max suppression.

        Args:
            filtered_boxes (numpy.ndarray): Filtered boxes.
            box_classes (numpy.ndarray): Box class indices.
            box_scores (numpy.ndarray): Box scores.

        Returns:
            tuple:
                (box_predictions,
                 predicted_box_classes,
                 predicted_box_scores)
        """
        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        unique_classes = np.unique(box_classes)

        for cls in unique_classes:
            idxs = np.where(box_classes == cls)

            cls_boxes = filtered_boxes[idxs]
            cls_classes = box_classes[idxs]
            cls_scores = box_scores[idxs]

            order = np.argsort(cls_scores)[::-1]

            cls_boxes = cls_boxes[order]
            cls_classes = cls_classes[order]
            cls_scores = cls_scores[order]

            while len(cls_scores) > 0:
                box_predictions.append(cls_boxes[0])
                predicted_box_classes.append(cls_classes[0])
                predicted_box_scores.append(cls_scores[0])

                if len(cls_scores) == 1:
                    break

                x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
                y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
                x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
                y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

                inter_w = np.maximum(0, x2 - x1)
                inter_h = np.maximum(0, y2 - y1)

                intersection = inter_w * inter_h

                area1 = ((cls_boxes[0, 2] - cls_boxes[0, 0]) *
                         (cls_boxes[0, 3] - cls_boxes[0, 1]))

                area2 = ((cls_boxes[1:, 2] - cls_boxes[1:, 0]) *
                         (cls_boxes[1:, 3] - cls_boxes[1:, 1]))

                union = area1 + area2 - intersection

                iou = intersection / union

                keep_idxs = np.where(iou < self.nms_t)[0]

                cls_boxes = cls_boxes[keep_idxs + 1]
                cls_classes = cls_classes[keep_idxs + 1]
                cls_scores = cls_scores[keep_idxs + 1]

        return (np.array(box_predictions),
                np.array(predicted_box_classes),
                np.array(predicted_box_scores))

    @staticmethod
    def load_images(folder_path):
        """
        Load all images from a folder.

        Args:
            folder_path (str): Path to folder containing images.

        Returns:
            tuple: (images, image_paths)
        """
        images = []
        image_paths = []

        for filename in os.listdir(folder_path):
            path = os.path.join(folder_path, filename)

            image = cv2.imread(path)

            if image is not None:
                images.append(image)
                image_paths.append(path)

        return (images, image_paths)

    def preprocess_images(self, images):
        """
        Preprocess images for Darknet model.

        Args:
            images (list): List of images as numpy.ndarrays.

        Returns:
            tuple: (pimages, image_shapes)
        """
        pimages = []
        image_shapes = []

        input_h = self.model.input.shape[1]
        input_w = self.model.input.shape[2]

        for image in images:
            image_shapes.append(image.shape[:2])

            resized = cv2.resize(
                image,
                (input_w, input_h),
                interpolation=cv2.INTER_CUBIC
            )

            resized = resized / 255.0

            pimages.append(resized)

        pimages = np.array(pimages)
        image_shapes = np.array(image_shapes)

        return (pimages, image_shapes)

    def show_boxes(self, image, boxes, box_classes,
                   box_scores, file_name):
        """
        Display image with boundary boxes.

        Args:
            image (numpy.ndarray): Original image.
            boxes (numpy.ndarray): Boundary boxes.
            box_classes (numpy.ndarray): Class indices.
            box_scores (numpy.ndarray): Box scores.
            file_name (str): Original image file name.
        """
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.astype(int)

            class_name = self.class_names[box_classes[i]]
            score = box_scores[i]

            text = "{} {:.2f}".format(class_name, score)

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2
            )

            cv2.putText(
                image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(file_name, image)

        key = cv2.waitKey(0)

        if key == ord('s'):
            if not os.path.exists("detections"):
                os.makedirs("detections")

            save_path = os.path.join("detections", file_name)

            cv2.imwrite(save_path, image)

        cv2.destroyAllWindows()
