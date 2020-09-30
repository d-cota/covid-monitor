import cv2

from abc import ABC, abstractmethod
from utils import fontColor, fontScale, font, lineType


class Detector(ABC):
    @abstractmethod
    def detect(self, image, targets='None'):
        """

                :param image: image to perform the inference on
                :param (list of str) targets: desired class targets, ex. 'person'
                :return (list of list): output structure [[x1, y1, x2, y2, confidence, class], ...]
                """
        pass


def count_people(image, bboxes, draw_rectangles=False):
    """

    :param image:
    :param bboxes:
    :param draw_rectangles:
    :return:
    """

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

    return image
