from abc import ABC, abstractmethod


class Detector(ABC):
    @abstractmethod
    def detect(self, image, targets='None'):
        """

                :param image: image to perform the inference on
                :param (list of str) targets: desired class targets, ex. 'person'
                :return (list of list): output in the form [x1, y1, x2, y2, confidence, class]
                """
        pass

    @abstractmethod
    def draw_boxes(self, image, bboxes):
        pass
