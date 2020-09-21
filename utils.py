import cv2


class ImageIterator(object):
    """
    Iterator class used to iterate over video frames
    """
    def __init__(self, frameGetter, frameRate=1, max_frames=None, skip_sec=0, resize=None):
        """

        :param frameGetter: usually a cv2 videoCapture object
        :param frameRate: how many frames for iteration
        :param max_frames: how many frames you want to visualize
        :param skip_sec: from which point in seconds you want to start getting the frames
        :param (tuple) resize: desired image dimension
        """
        self.frameGetter = frameGetter
        self.frameRate = frameRate
        self.max_frames = max_frames
        self.sec = skip_sec
        self.resize = resize

    def __iter__(self):
        self.count = 0
        self.frameGetter.set(cv2.CAP_PROP_POS_MSEC, self.sec * 1000)
        return self

    def __next__(self):
        for i in range(self.frameRate):
            hasFrames, image = self.frameGetter.read()
        self.count += 1
        if hasFrames is False or (self.max_frames is not None and self.count > self.max_frames):
            raise StopIteration
        else:
            if self.resize is None:
                return image
            else:
                return cv2.resize(image, self.resize)