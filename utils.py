import cv2
import argparse

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2


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


def convertToXYWH(x1, y1, x2, y2):
    """
    Transforms rectangular coordinates into center coordinates
    :param x1: x min
    :param y1: y min
    :param x2: x max
    :param y2: y max
    :return (list): rectangle center coordinates [x, y, w, h]
                    with x center, y center, width, height
    """
    x = int((x1 + x2) / 2)
    y = int((y1 + y2) / 2)
    w = int(x2 - x1)
    h = int(y2 - y1)

    return [x, y, w, h]


def convertToXYXY(x, y, w, h):
    """
    Transforms rectangular coordinates into center coordinates
    :param x: x center
    :param y: y center
    :param w: width
    :param h: height
    :return (list):
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))

    return xmin, ymin, xmax, ymax


def FPS(image, fps):
    topRightCorner = (int(image.shape[1] * 0.77), 30)
    image = cv2.putText(image, 'FPS: {:.2f}'.format(fps),
                        topRightCorner,
                        font,
                        fontScale,
                        fontColor,
                        lineType)
    return image


def read_arguments():
    parser = argparse.ArgumentParser("Perform inference to monitor covid prevention measures.")
    parser.add_argument("--detector_weights", type=str, default='./weights/yolov4.pth',
                        help='detector weights file path',
                        dest='det_weights')
    parser.add_argument("--names", type=str, default='./config/coco.names', help='classes file path',
                        dest='names')
    parser.add_argument("--video", type=str, help="video path to perform inference on", dest='video')
    parser.add_argument("--camera", type=str, help="if your input comes from a videocamera, put the camera number ("
                                                   "usually 0)", dest='camera')
    parser.add_argument("--dim", type=int, nargs='+', default=(320, 320), help='dimension of the net a.k.a. dimension '
                                                                               'of the '
                                                                               'input image. It must be in the form of '
                                                                               '(x, '
                                                                               'y) where x or y can be defined as: x = '
                                                                               '320 + 96 '
                                                                               '* n in {0, 1, 2, ...} ', dest='dim')
    parser.add_argument("-d", "--distancing", action='store_true', help='enable social distancing monitor')
    parser.add_argument("-dt", "--distancing_tracking", action='store_true', help='enable social distancing monitor '
                                                                                  'with tracking', dest='dist_track')
    parser.add_argument("-m", "--mask", action='store_true', help='enable mask detection')
    parser.add_argument("-mt", "--mask_tracking", action='store_true', help='enable mask detection with tracking',
                        dest='mask_track')
    parser.add_argument("--dist_thresh", type=float, default=2, help='minimum distance in meters to respect social '
                                                                     'distancing', dest='thresh')
    parser.add_argument("--root_weights", type=str, default='./weights/snapshot_19.pth.tar', help='rootnet snapshot '
                                                                                                  'file path',
                        dest='rootPath')
    parser.add_argument("--focal", type=int, nargs='+', default=(1500, 1500), help='rootnet focal lenght parameter',
                        dest='focal')
    parser.add_argument("--resize", type=int, nargs='+', default=(1080, 720), help='output video dimension')
    parser.add_argument("--hpe_weights", type=str, default='./weights/pose_hrnet_w48_384x288.pth',
                        help='human pose estimator weights file path',
                        dest='hpePath')
    parser.add_argument("--classifier_weights", type=str, default='./weights/bit-101x1.pth',
                        help='mask classifier weights file path',
                        dest='clfPath')
    parser.add_argument("--tracker_weights", type=str, default='./weights/ckpt.t7',
                        help='tracker weights file path',
                        dest='trackPath')
    parser.add_argument("--framerate", type=int, default=1, help='how many frames you want to skip',
                        dest='framerate')
    parser.add_argument("--save", type=str, help='filename output video you want to save. All files will be saved in '
                                                 'the outputs folder', dest='saveName')
    parser.add_argument("--eyes_thresh", type=float, default=0.8, help='Minimum eyes confidence threshold to be '
                                                                       'considered a valid face detection')
    parser.add_argument("--mask_thresh", type=int, default=80, help='Minimum confidence threshold to accept the mask '
                                                                    'prediction')
    parser.add_argument("--len_buffer", type=int, default=21, help='length of the buffer containing the mask '
                                                                   'predictions')
    parser.add_argument("--last_frames", type=int, default=3, help='Number of frames to compute the mean on for the '
                                                                   'social distancing monitoring (only valid with '
                                                                   'tracking on)')
    parser.add_argument("--skeleton", action='store_true', help='enable mask detection')
    return parser.parse_args()
