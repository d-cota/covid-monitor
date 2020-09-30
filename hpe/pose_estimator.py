from abc import ABC, abstractmethod


class HPEstimator(ABC):
    @abstractmethod
    def predict(self, detections, image):
        """

        :param detections:
        :param image:
        :return:
        """

        pass
