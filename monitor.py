import time
import cv2
import sys
import argparse

from detector.yolov4.model import YOLOv4
from detector.detector import count_people
from tracking.deep_sort import DeepSort
from rootnet.model import RootNet, check_distance

from utils import ImageIterator


def read_arguments():
    parser = argparse.ArgumentParser("Perform inference to monitor covid prevention measures.")
    parser.add_argument("--weights", type=str, default='./weights/yolov4.pth', help='.pth weights file path',
                        dest='weights')
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
    parser.add_argument("--tracking", help='enable tracking')
    parser.add_argument("--root_weights", type=str, default='./weights/snapshot_19.pth.tar', help='rootnet snapshot '
                                                                                                  'file path',
                        dest='rootPath')
    parser.add_argument("--focal", type=int, nargs='+', default=(1500, 1500), help='rootnet focal lenght parameter', dest='focal')
    return parser.parse_args()


args = read_arguments()
if not args.video and not args.camera:
    sys.exit("Please insert a valid input.")
elif not args.video:
    input_video = int(args.camera)
else:
    input_video = args.video
if not args.weights:
    sys.exit("Please insert a valid weights path.")
if not args.names:
    sys.exit("Please insert a valid names path.")
if not args.dim:
    sys.exit("Please insert a valid dimension value.")

weights_path = args.weights
names_path = args.names
dim = tuple(args.dim)
tracker_weights = './weights/ckpt.t7'

detector = YOLOv4(weights_path, names_path, dim)
if args.tracking is not None:
    tracker = DeepSort(tracker_weights)
rootNet = RootNet(args.rootPath, focal=args.focal)
imageIterator = ImageIterator(cv2.VideoCapture(input_video), resize=(1080, 720), frameRate=4)

for image in imageIterator:
    start = time.time()
    bboxes = detector.detect(image, ['person'])

    if args.tracking is not None:
        outputs = tracker.update(bboxes, image)

    outputs = rootNet.estimate(bboxes, image)

    output_image = count_people(image, bboxes)  # detector image
    # output_image = tracker.draw_boxes(image, outputs, start)  # tracker image
    output_image = check_distance(output_image, outputs, start)  # rootnet image
    cv2.waitKey(1)
    cv2.imshow('COVID-monitor', output_image)

    fps = 1. / (time.time() - start)
    print('\rframerate: %f fps' % fps, end='')
