import time

import torch
import torchvision as tv
import cv2
from PIL import Image
from hpe.simplehrnet.visualization import get_face
from collections import deque, Counter
from utils import fontColor, fontScale, font, lineType

LABELS = ['mask', 'no_mask']


class BiTClassifier(object):
    def __init__(self, weightsPath):
        self.model = torch.load(weightsPath)
        self.model.eval()

        self.transform = tv.transforms.Compose([
            tv.transforms.Resize((128, 128)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def predict(self, face_image):
        img = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = self.transform(img)
        with torch.no_grad():
            img = img.to('cuda', non_blocking=True)
            logits = self.model(img[None, ...])
            p = torch.nn.functional.softmax(logits, dim=1)
            label = torch.argmax(p)
            label = label.item()
            score = p[0][label].item()

            return label, int(score * 100)


def detect_mask(clf, hpe_list, image, out_image, eyes_thresh=0.8, conf_thresh=80, max_len=21, tracking=False):
    bboxes, pts = hpe_list[:2]
    nomask_counter = 0
    if tracking:
        person_ids = hpe_list[-1]
        for i, (pt, bbox, pid) in enumerate(zip(pts, bboxes, person_ids)):
            face_img = get_face(image, pt, eyes_threshold=eyes_thresh)
            if pid not in detect_mask.votes:
                detect_mask.votes[pid] = deque(maxlen=max_len)
            if face_img is not None:
                label, score = clf.predict(face_img)
                if score >= conf_thresh:
                    detect_mask.votes[pid].append(LABELS[label])
            most_freq = Counter(detect_mask.votes[pid]).most_common()
            if len(most_freq) == 0:  # there are no valid predictions yet
                continue
            freq_class, occurrences = most_freq[0]
            if occurrences > 3:
                if freq_class == 'mask':
                    color = (0, 255, 0)  # green for mask
                else:
                    color = (0, 0, 255)  # red for no_mask
                    nomask_counter += 1
                cv2.putText(out_image, "(" + str(pid) + ") " + freq_class + " - " + str(occurrences),
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        for i, (pt, bbox) in enumerate(zip(pts, bboxes)):
            face_img = get_face(image, pt, eyes_threshold=eyes_thresh)
            if face_img is None:
                continue
            label, score = clf.predict(face_img)
            if score >= conf_thresh:
                if label:
                    color = (0, 0, 255)
                    nomask_counter += 1
                else:
                    color = (0, 255, 0)

                cv2.putText(out_image, LABELS[label] + " - " + str(score),
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    bottomLeftCornerOfText = (10, int(image.shape[0] * 0.86))  # width, height
    out_image = cv2.putText(out_image, 'People not wearing mask: {}'.format(nomask_counter),
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineType)


    return out_image


detect_mask.votes = {}
