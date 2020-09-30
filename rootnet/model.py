import math
import time

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn.parallel.data_parallel import DataParallel

from .main.config import cfg
from .main.model import get_pose_net
from .common.utils.pose_utils import process_bbox, pixel2cam
from .data.dataset import generate_patch_image

from utils import convertToXYWH, fontColor, fontScale, font, lineType
from itertools import combinations
from collections import deque


class RootNet(object):

    def __init__(self, weightsPath, principal_points=None, focal=(1500, 1500)):
        """

        :param weightsPath:
        :param principal_points:
        :param focal:
        """

        self.focal = focal
        self.principal_points = principal_points

        self.net = get_pose_net(cfg, False)
        self.net = DataParallel(self.net).cuda()
        weigths = torch.load(weightsPath)
        self.net.load_state_dict(weigths['network'])
        self.net.eval()

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])

    def estimate(self, bboxes, image, tracking=False):
        """

        :param bboxes:
        :param image:
        :return:
        """
        if self.principal_points is None:
            self.principal_points = [image.shape[1] / 2, image.shape[0] / 2]

        output = []
        for bbox in bboxes:
            bbox_xywh = convertToXYWH(bbox[0], bbox[1], bbox[2], bbox[3])
            bbox_root = process_bbox(bbox_xywh, image.shape[1], image.shape[0])
            img, img2bb_trans = generate_patch_image(image, bbox_root, False, 0.0)
            img = self.transform(img).cuda()[None, :, :, :]
            k_value = np.array([math.sqrt(cfg.bbox_real[0] * cfg.bbox_real[1] * self.focal[0] * self.focal[1] / (
                    bbox_root[2] * bbox_root[3]))]).astype(np.float32)
            k_value = torch.FloatTensor([k_value]).cuda()[None, :]

            # forward
            with torch.no_grad():
                root_3d = self.net(img, k_value)  # x,y: pixel, z: root-relative depth (mm)
            root_3d = root_3d[0].cpu().numpy()

            # inverse affine transform (restore the crop and resize)
            root_3d[0] = root_3d[0] / cfg.output_shape[1] * cfg.input_shape[1]
            root_3d[1] = root_3d[1] / cfg.output_shape[0] * cfg.input_shape[0]
            root_3d_xy1 = np.concatenate((root_3d[:2], np.ones_like(root_3d[:1])))
            img2bb_trans_001 = np.concatenate((img2bb_trans, np.array([0, 0, 1]).reshape(1, 3)))
            root_3d[:2] = np.dot(np.linalg.inv(img2bb_trans_001), root_3d_xy1)[:2]
            # get 3D coordinates for bbox
            root_3d = pixel2cam(root_3d[None, :], self.focal, self.principal_points)
            if tracking:
                pid = bbox[-1]
                output.append([bbox, root_3d, pid])
            else:
                output.append([bbox, root_3d])

        return output


def check_distance(image, outputs, threshold=2, last_frames=3, tracking=False):
    """

    :param last_frames:
    :param threshold:
    :param start:
    :param image:
    :param outputs:
    :return:
    """
    centroid_dict = dict()
    objectId = 0
    for out in outputs:
        bbox = out[0]
        coord = out[1]
        if tracking:  # if we're using tracking
            pid = out[-1]
            if pid not in check_distance.coordinates:
                check_distance.coordinates[pid] = deque(maxlen=last_frames)
            check_distance.coordinates[pid].append(coord)
            if len(check_distance.coordinates[pid]) >= last_frames:  # if the buffer is full
                c = np.array(check_distance.coordinates[pid])
                mean_coord = np.mean(c, axis=0)  # compute the mean of the last n frames
                centroid_dict[objectId] = (
                mean_coord[0][0], mean_coord[0][1], mean_coord[0][2], bbox[0], bbox[1], bbox[2], bbox[3])
                objectId += 1
            else:
                continue
        else:  # if we're not using tracking, just take the raw coordinates
            centroid_dict[objectId] = (coord[0][0], coord[0][1], coord[0][2], bbox[0], bbox[1], bbox[2], bbox[3])
            objectId += 1

    red_zone_list = []  # List containing which Object id is in under threshold distance condition.
    for (id1, p1), (id2, p2) in combinations(centroid_dict.items(), 2):
        dx, dy, dz = p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)  # Calculates the Euclidean distance
        if distance < (threshold * 1000):  # Set our social distance threshold
            if id1 not in red_zone_list:
                red_zone_list.append(id1)
            if id2 not in red_zone_list:
                red_zone_list.append(id2)

    for idx, box in centroid_dict.items():  # dict (1(key):red(value), 2 blue)  idx - key  box - value
        if idx in red_zone_list:  # if id is in red zone list
            cv2.rectangle(image, (box[3], box[4]), (box[5], box[6]), (0, 0, 255), 1)  # Create Red bounding boxes

    text = "People at Risk: %s" % str(len(red_zone_list))  # Count People at Risk
    bottomLeftCornerOfText = (10, int(image.shape[0] * 0.92))  # width, height
    image = cv2.putText(image, text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        lineType)

    return image


check_distance.coordinates = {}
