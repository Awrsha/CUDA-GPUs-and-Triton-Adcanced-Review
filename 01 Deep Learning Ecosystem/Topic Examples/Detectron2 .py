import torch
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# Set up Detectron2 configuration for a pre-trained model
# The `get_cfg()` function loads the default configuration from Detectron2, which we will then modify.
cfg = get_cfg()

# Merge the configuration file for a pre-trained model from the Detectron2 model zoo.
# Here, we load a model for instance segmentation based on Mask R-CNN with a ResNet-50 backbone and FPN (Feature Pyramid Networks).
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

# Set the path to the pre-trained model weights (downloaded from the model zoo).
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

# Set the threshold for instance detection. Only detections with a score higher than 0.5 will be shown.
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Minimum score threshold for detection

# Automatically select the device based on availability (GPU or CPU).
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create a predictor for inference using the defined configuration.
# `DefaultPredictor` loads the model based on the provided configuration and runs inference on input images.
predictor = DefaultPredictor(cfg)

# Import the necessary libraries for image loading and processing
from detectron2.data.datasets import register_coco_instances
import cv2

# Load an input image using OpenCV
im = cv2.imread("example.jpg")

# Perform inference with the predictor on the input image
# `predictor(im)` runs the model on the image and returns a dictionary containing the detected instances.
outputs = predictor(im)

# Visualize the results with the `Visualizer` class.
# The `Visualizer` will overlay the predicted instances (such as bounding boxes, masks, and labels) on the image.
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
# The `im[:, :, ::-1]` converts the image from BGR (OpenCV default) to RGB (required by the Visualizer).
# `MetadataCatalog.get(cfg.DATASETS.TRAIN[0])` fetches metadata for the dataset, such as category names, colors, etc.

# Draw the instance predictions on the image.
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# `outputs["instances"].to("cpu")` moves the instances to CPU for visualization (if they were on GPU).

# Display the final image with detections using OpenCV.
cv2.imshow("Detection Output", out.get_image()[:, :, ::-1])
# `out.get_image()` retrieves the image with drawn predictions in RGB format.
# `[:, :, ::-1]` converts it back to BGR for OpenCV display.
cv2.waitKey(0)  # Wait for a key press before closing the image window