# Monitoring COVID-19 prevention measures on CCTV cameras using Deep Learning
This repo contains the code for the thesis *"Monitoring COVID-19 prevention measures on CCTV cameras using Deep Learning"*. It is divided into three sections, one for each task.
## Project structure
The `${COVID_MONITOR}` is described as below.
```
${COVID_MONITOR}
|-- config
|-- inputs
|-- outputs
|-- weights
|-- detector
|-- utils.py
```
* `config` contains configuration files for the models.
* `inputs` contains video and images to process.
* `outputs` contains resulting outputs.
* `weigths` contains the model checkpoints to be loaded.
* `detector` contains the code for object detection algorithm
* `utils.py` contains some utilities functions.

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

    @abstractmethod
    def draw_boxes(self, image, bboxes):
        pass
```


## People counting
The system relies on a object detector to perform the people counting task. To start counting people launch [monitor.py](https://github.com/d-cota/covid-monitor/blob/master/monitor.py) with the following parameters:
```
--weights: pth weights file path
--names: classes file path
--video: video path to perform inference on
--camera: if your input comes from a videocamera, put the camera number (usually 0)
--dim: dimension of the net a.k.a. dimension of the input image. It must be in the form of (x, y) where x or y can be defined as (only valid for YOLO): x = 320 + 96 * n in {0, 1, 2, ...}
```
*Usage example*
```
$ python monitor.py --weights ./weights/yolov4.pth --names ./config/coco.names --video ./inputs/demo.mp4 --dim 320 320
```
You can download YOLOv4 weights from [here](https://drive.google.com/open?id=1wv_LiFeCRYwtpkqREPeI13-gPELBDwuJ) and put them in the weights directory.
