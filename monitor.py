import warnings
import time
import cv2

from detector.yolov4.model import YOLOv4
from detector.detector import count_people
from tracking.deep_sort import DeepSort
from rootnet.model import RootNet, check_distance
from hpe.simplehrnet.model import SimpleHRNet
from hpe.simplehrnet.visualization import draw
from classifier.bit import BiTClassifier, detect_mask

from utils import ImageIterator, FPS, read_arguments


warnings.filterwarnings("ignore", category=DeprecationWarning)
args = read_arguments()

dim = tuple(args.dim)
resize = tuple(args.resize)

if len(args.video) < 3:  # if the source is the camera
    src = int(args.video)
else:
    src = args.video

detector = YOLOv4(args.det_weights, args.names, dim)

if args.dist_track or args.mask_track:
    tracker = DeepSort(args.trackPath)

if args.distancing or args.dist_track:
    rootNet = RootNet(args.rootPath, focal=args.focal)
    distance = args.thresh

if args.mask or args.mask_track:
    hpEstimator = SimpleHRNet(c=48, nof_joints=17, checkpoint_path=args.hpePath)
    classifier = BiTClassifier(args.clfPath)

imageIterator = ImageIterator(cv2.VideoCapture(src), resize=resize, frameRate=args.framerate)

if args.saveName is not None:
    videoWriter = cv2.VideoWriter(args.saveName, cv2.VideoWriter_fourcc(*'XVID'), 15.0, resize)

for image in imageIterator:
    start = time.time()
    bboxes = detector.detect(image, ['person'])  # [[x1, y1, x2, y2, conf, class], ...]
    output_image = count_people(image, bboxes)  # detector image

    if args.dist_track or args.mask_track:
        bboxes = tracker.update(bboxes, image)  # [[x1, y1, x2, y2, id], ...]
        # output_image = tracker.draw_boxes(image, bboxes)  # tracker image

    if args.distancing or args.dist_track:
        root_output = rootNet.estimate(bboxes, image, tracking=args.dist_track)  # [[x1, y1, x2, y2, x3D, y3D, z3D],...]
        output_image = check_distance(output_image, root_output, threshold=distance, last_frames=args.last_frames,
                                      tracking=args.dist_track)  # rootnet image

    if args.mask or args.mask_track:
        hpe_output = hpEstimator.predict(bboxes, image)  # [[x1, y1, x2, y2], ...], [[joints], ...], [pid, ...]]
        if args.skeleton:
            output_image = draw(hpe_output[1], image)  # skeleton image
        output_image = detect_mask(classifier, hpe_output, image, output_image, conf_thresh=args.mask_thresh,
                                   eyes_thresh=args.eyes_thresh, max_len=args.len_buffer, tracking=args.mask_track)

    fps = 1. / (time.time() - start)
    output_image = FPS(output_image, fps)
    cv2.waitKey(1)
    cv2.imshow('COVID-monitor', output_image)

    if args.saveName is not None:
        videoWriter.write(output_image)

    print('\rframerate: %f fps' % fps, end='')
