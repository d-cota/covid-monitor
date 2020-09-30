import numpy as np
import torch
import cv2

from torchvision import transforms
from ..pose_estimator import HPEstimator
from .models.hrnet import HRNet
from .models.poseresnet import PoseResNet


class SimpleHRNet(HPEstimator):
    """
        SimpleHRNet class.

        The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
        weights, and predict the human pose on single images.
        Multi-person support with the YOLOv3 detector is also included (and enabled by default).
        """

    def __init__(self,
                 c,
                 nof_joints,
                 checkpoint_path,
                 model_name='HRNet',
                 resolution=(384, 288),
                 interpolation=cv2.INTER_CUBIC,
                 device=torch.device("cuda")):
        """
        Initializes a new SimpleHRNet object.
        HRNet is initialized on the torch.device("device") and
        its pre-trained weights will be loaded from disk.

        Args:
            c (int): number of channels (when using HRNet model) or resnet size (when using PoseResNet model).
            nof_joints (int): number of joints.
            checkpoint_path (str): path to an official hrnet checkpoint or a checkpoint obtained with `train_coco.py`.
            model_name (str): model name (HRNet or PoseResNet).
                Valid names for HRNet are: `HRNet`, `hrnet`
                Valid names for PoseResNet are: `PoseResNet`, `poseresnet`, `ResNet`, `resnet`
                Default: "HRNet"
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: (384, 288)
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: torch.device("cpu")
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.device = device

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.resolution[0], self.resolution[1])),  # (height, width)
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if model_name in ('HRNet', 'hrnet'):
            self.model = HRNet(c=c, nof_joints=nof_joints)
        elif model_name in ('PoseResNet', 'poseresnet', 'ResNet', 'resnet'):
            self.model = PoseResNet(resnet_size=c, nof_joints=nof_joints)
        else:
            raise ValueError('Wrong model name.')

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])
        else:
            self.model.load_state_dict(checkpoint)

        if 'cuda' in str(self.device):
            print("device: 'cuda' - ", end="")

            if 'cuda' == str(self.device):
                # if device is set to 'cuda', all available GPUs will be used
                print("%d GPU(s) will be used" % torch.cuda.device_count())
                device_ids = None
            else:
                # if device is set to 'cuda:IDS', only that/those device(s) will be used
                print("GPU(s) '%s' will be used" % str(self.device))
                device_ids = [int(x) for x in str(self.device)[5:].split(',')]

            self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        elif 'cpu' == str(self.device):
            print("device: 'cpu'")
        else:
            raise ValueError('Wrong device name.')

        self.model = self.model.to(device)
        self.model.eval()

    def predict(self, detections, image):
        nof_people = len(detections)
        boxes = np.empty((nof_people, 4), dtype=np.int32)
        if nof_people > 0 and len(detections[0]) == 5:  # if detections include person-id, i.e. we're using tracking
            person_ids = np.empty(nof_people, dtype=np.int32)
        else:
            person_ids = None
        images = torch.empty((nof_people, 3, self.resolution[0], self.resolution[1]))  # (height, width)

        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d[:4]
            if person_ids is not None:
                pid = d[4]
            # Adapt detections to match HRNet input aspect ratio (as suggested by xtyDoge in issue #14)
            correction_factor = self.resolution[0] / self.resolution[1] * (x2 - x1) / (y2 - y1)
            if correction_factor > 1:
                # increase y side
                center = y1 + (y2 - y1) // 2
                length = int(round((y2 - y1) * correction_factor))
                y1 = max(0, center - length // 2)
                y2 = min(image.shape[0], center + length // 2)
            elif correction_factor < 1:
                # increase x side
                center = x1 + (x2 - x1) // 2
                length = int(round((x2 - x1) * 1 / correction_factor))
                x1 = max(0, center - length // 2)
                x2 = min(image.shape[1], center + length // 2)

            try:
                boxes[i] = [x1, y1, x2, y2]
                images[i] = self.transform(image[y1:y2, x1:x2, ::-1])
                if person_ids is not None:
                    person_ids[i] = pid
            except ValueError:
                i -= 1
                continue

        if images.shape[0] > 0:
            images = images.to(self.device)

            with torch.no_grad():
                out = self.model(images)

            out = out.detach().cpu().numpy()
            pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
            # For each human, for each joint: y, x, confidence
            for i, human in enumerate(out):
                for j, joint in enumerate(human):
                    pt = np.unravel_index(np.argmax(joint), (self.resolution[0] // 4, self.resolution[1] // 4))
                    # 0: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                    # 1: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                    # 2: confidences
                    pts[i, j, 0] = pt[0] * 1. / (self.resolution[0] // 4) * (boxes[i][3] - boxes[i][1]) + boxes[i][1]
                    pts[i, j, 1] = pt[1] * 1. / (self.resolution[1] // 4) * (boxes[i][2] - boxes[i][0]) + boxes[i][0]
                    pts[i, j, 2] = joint[pt]

        else:
            pts = np.empty((0, 0, 3), dtype=np.float32)

        if person_ids is not None:
            return [boxes, pts, person_ids]
        else:
            return [boxes, pts]

