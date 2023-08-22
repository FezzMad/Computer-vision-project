import cv2
import numpy as np


def img_preparation(img, imgsz):
    """
    Preparing an image to be fed to the input of a neural network:
    change color space [BGR -> RGB]
    change data type [uint8 -> float23]
    normalization [0..255 -> 0..1]
    change order [W,H,C -> C,W,H]

    :param img: (numpy.ndarray) image for preparation from cv2.imread
    :param imgsz: (int) image size, width or height
    :return: (numpy.ndarray) prepared image
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change color space [BGR -> RGB]
    img = img.astype(np.float32)  # change data type [uint8 -> float23]
    img = img / 255.0  # normalization [0..255 -> 0..1]
    img = cv2.resize(img, (imgsz, imgsz), cv2.INTER_AREA)
    img = img.transpose((2, 0, 1))  # change order [width, height, channels -> channels, width, height]
    # [ (0),(1),(2) -> (2),(0),(1) ]
    return img