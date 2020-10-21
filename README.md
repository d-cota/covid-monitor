![madeby](https://img.shields.io/badge/made%20by-d--cota-blue)
![GitHub top language](https://img.shields.io/github/languages/top/d-cota/covid-monitor)
![GitHub Repo stars](https://img.shields.io/github/stars/d-cota/covid-monitor?color=yellow)
![GitHub issues](https://img.shields.io/github/issues/d-cota/covid-monitor)
![GitHub pull requests](https://img.shields.io/github/issues-pr/d-cota/covid-monitor)

# Monitoring COVID-19 prevention measures on CCTV cameras using Deep Learning
This repo contains the code for the thesis [*"Monitoring COVID-19 prevention measures on CCTV cameras using Deep Learning"*](https://www.researchgate.net/publication/344785970_Monitoring_COVID-19_prevention_measures_on_CCTV_cameras_using_Deep_Learning). The whole system is implemented in PyTorch and has been tested on Windows 10. Technical slide can be found [here](https://github.com/d-cota/covid-monitor/blob/master/technical%20slides.pdf).

<p align="center">
  <img width="600" height="400" src="https://github.com/d-cota/covid-monitor/blob/master/example.gif">
</p>

## Project structure
The `${COVID_MONITOR}` is described as below.
```
${COVID_MONITOR}
|-- bit_pytorch
|-- classifier
|-- config
|-- detector
|-- inputs
|-- hpe
|-- outputs
|-- rootnet
|-- tracking
|-- weights
|-- monitor.py
|-- utils.py
```
* `bit_pytorch` contains the ResNet [Big Transfer](https://github.com/google-research/big_transfer) implementation for the classifier.
* `config` contains configuration files for the models.
* `detector` contains the code for the object detection algorithm.
* `inputs` contains video and images to process.
* `hpe` contains the human pose estimation modules, in particular the [SimpleHRNet](https://github.com/stefanopini/simple-HRNet) implementation.
* `outputs` contains resulting outputs.
* `rootnet` contains the [RootNet](https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE) implementation for monitoring the social distancing
* `tracking` contains the code for the tracking algorithm, in particular the [DeepSort](https://github.com/kimyoon-young/centerNet-deep-sort) implementation
* `weigths` contains the model checkpoints to be loaded.
* `monitor.py` is the main script for monitoring the prevention measures.
* `utils.py` contains some utilities functions.

## Install

### Prerequisites
    * Python 3.7
    * PyTorch >= 1.4.0
    * CUDA >= 10
    * GIT
    * anaconda3
    * pip
### Installation
Please, open a shell and run the following command.
```sh
$ conda env create --file environment.yml
```

## Usage
Here are explained the various monitor.py custom parameters.
```
usage: Perform inference to monitor covid prevention measures.
       [-h] [--classifier_weights CLFPATH] [--detector_weights DET_WEIGHTS]
       [--dim DIM [DIM ...]] [-d] [--dist_thresh THRESH] [-dt]
       [--eyes_thresh EYES_THRESH] [--focal FOCAL [FOCAL ...]] [-f FRAMERATE]
       [--hpe_weights HPEPATH] [--last_frames LAST_FRAMES]
       [--len_buffer LEN_BUFFER] [-m] [--mask_thresh MASK_THRESH] [-mt]
       [--names NAMES] [-r RESIZE [RESIZE ...]] [--root_weights ROOTPATH]
       [-s SAVENAME] [--skeleton] [--tracker_weights TRACKPATH]
       video

positional arguments:
  video                 Video path to perform inference on. If you want to use
                        the camera, insert the corresponding camera number
                        (usually 0)

optional arguments:
  -h, --help            show this help message and exit
  --classifier_weights CLFPATH
                        mask classifier weights file path
  --detector_weights DET_WEIGHTS
                        detector weights file path
  --dim DIM [DIM ...]   dimension of the net a.k.a. dimension of the input
                        image. It must be in the form of (x, y) where x or y
                        can be defined as: x = 320 + 96 * n in {0, 1, 2, ...}
  -d, --distancing      enable social distancing monitor
  --dist_thresh THRESH  minimum distance in meters to respect social
                        distancing
  -dt, --distancing_tracking
                        enable social distancing monitor with tracking
  --eyes_thresh EYES_THRESH
                        Minimum eyes confidence threshold to be considered a
                        valid face detection
  --focal FOCAL [FOCAL ...]
                        rootnet focal lenght parameter
  -f FRAMERATE, --framerate FRAMERATE
                        how many frames you want to skip
  --hpe_weights HPEPATH
                        human pose estimator weights file path
  --last_frames LAST_FRAMES
                        Number of frames to compute the mean on for the social
                        distancing monitoring (only valid with tracking on)
  --len_buffer LEN_BUFFER
                        length of the buffer containing the mask predictions
  -m, --mask            enable mask detection
  --mask_thresh MASK_THRESH
                        Minimum confidence threshold to accept the mask
                        prediction
  -mt, --mask_tracking  enable mask detection with tracking
  --names NAMES         classes file path
  -r RESIZE [RESIZE ...], --resize RESIZE [RESIZE ...]
                        output video dimension
  --root_weights ROOTPATH
                        rootnet snapshot file path
  -s SAVENAME, --save SAVENAME
                        filename output video you want to save. All files will
                        be saved in the outputs folder
  --skeleton            enable mask detection
  --tracker_weights TRACKPATH
                        tracker weights file path

```

### People counting

*Usage example*
```
$ python monitor.py  ./inputs/demo.mp4 --detector_weights ./weights/yolov4.pth --names ./config/coco.names --dim 320 320
```

### Distancing monitoring

*Usage example*
```
$ python monitor.py  ./inputs/demo.mp4 -d --root_weights ./weights/snapshot_19.pth.tar --dist_thresh 2
```

### Mask detection with tracking

*Usage example*
```
$ python monitor.py  ./inputs/demo.mp4 -mt --hpe_weights ./weights/pose_hrnet_w48_384x288.pth --mask_thresh 80
```

### detector
The detector package is organized as follows:
```
${COVID_MONITOR}
|-- detector
|   |-- __init__.py
|   |-- detector.py
|   |   |-- yolov4
```
where [detector.py](https://github.com/d-cota/covid-monitor/blob/master/detector/detector.py) contains an interface to be implemented with the object detector you desire. We used [YOLOv4 PyTorch implementations](https://github.com/Tianxiaomo/pytorch-YOLOv4).
```python
class Detector(ABC):
    @abstractmethod
    def detect(self, image, targets='None'):
        """
                :param image: image to perform the inference on
                :param (list of str) targets: desired class targets, ex. 'person'
                :return (list of list): output in the form [x1, y1, x2, y2, confidence, class]
                """
        pass
```
## weights
Download the following files and put them in the weights directory.
* [YOLOv4 weights](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ)
* [DeepSort tracking checkpoints](https://github.com/kimyoon-young/centerNet-deep-sort/tree/master/deep/checkpoint)
* [Pretrained RootNet](https://drive.google.com/drive/folders/1nQfOIgc7_AG5xPAO-vtG_L0WxdOelxet?usp=sharing). Search for the Human3.6M+MPII folder, then take the snapshot_19.pth.tar file.
* [Pretrained PoseHRNet](https://drive.google.com/drive/folders/1nzM_OBV9LbAEA7HClC0chEyf_7ECDXYA). Download the pose_hrnet_w48_384x288.pth file.
* [Fine-tuned BiT-R101x1](https://drive.google.com/file/d/16G_sTX1WxM3Tw9u1uiYxYA6LpTvNKS0k/view?usp=sharing)

## Citation
If you liked this work please put a star and cite me :sunglasses:
```
DOI: 10.13140/RG.2.2.16368.69124
```

## References
This work is based on the following repositories:
* pytorch-YOLOv4 https://github.com/Tianxiaomo/pytorch-YOLOv4
* centerNet-deep-sort https://github.com/kimyoon-young/centerNet-deep-sort
* simple-HRNet https://github.com/stefanopini/simple-HRNet
* 3DMPPE_ROOTNET_RELEASE https://github.com/mks0601/3DMPPE_ROOTNET_RELEASE
* big_transfer https://github.com/google-research/big_transfer
