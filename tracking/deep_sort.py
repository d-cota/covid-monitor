import time

import cv2
import numpy as np

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker

from utils import convertToXYWH, fontColor, fontScale, font, lineType


class DeepSort(object):
    def __init__(self, model_path):
        self.min_confidence = 0.3
        self.nms_max_overlap = 1.0

        self.extractor = Extractor(model_path, use_cuda=True)

        max_cosine_distance = 0.2
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, bbox_xyxy, ori_img):
        """

        :param bbox_xyxy:
        :param ori_img:
        :return:
        """
        bbox_xywh = [convertToXYWH(bbox[0], bbox[1], bbox[2], bbox[3]) for bbox in bbox_xyxy]
        confidences = [c[-2] for c in bbox_xyxy]
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        try:
            features = self._get_features(bbox_xywh, ori_img)
        except:
            print('a')
        detections = [Detection(bbox_xywh[i], conf, features[i]) for i, conf in enumerate(confidences) if
                      conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._xywh_to_xyxy_yolo(box)
            track_id = track.track_id
            outputs.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs

    # for yolo  (centerx,centerx, w,h -> x1,y1,x2,y2)
    def _xywh_to_xyxy_yolo(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        features = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy_yolo(box)
            im = ori_img[y1:y2, x1:x2]
            feature = self.extractor(im)[0]
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features

    def draw_boxes(self, image, bboxes, draw_rectangles=True):
        people_count = len(bboxes)
        bottomLeftCornerOfText = (10, int(image.shape[0] * 0.98))  # width, height

        image = cv2.putText(image, 'People count: {}'.format(people_count),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)

        if draw_rectangles:
            for bbox in bboxes:
                image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
                image = cv2.putText(image, str(bbox[4]), (bbox[0], bbox[3] - 10), font, fontScale, fontColor, lineType)

        return image
