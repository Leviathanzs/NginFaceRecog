import sys
import os
import logging
import cv2
import torch
import base64
import io
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

from infranlib.align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from infranlib.align.detector import detect_faces
from infranlib.align.get_nets import ONet, RNet, PNet

import infranlib.databases.redis_handler as IMQ
import infranlib.databases.pool_handler as WKP

reference = None
input_size = None
crop_size = None
pnet = None
rnet = None
onet = None


def get_reference():

    global reference
    global input_size
    global crop_size

    input_size = [112, 112]
    crop_size = 112
    scale = crop_size / 112.0
    reference = get_reference_facial_points(default_square=True) * scale

def mtcnn_detector():

    global pnet
    global rnet
    global onet

    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()

def face_alignment(image_data):

    global reference
    global crop_size
    global pnet
    global onet
    global rnet

    img_warp = None
    img_str = None
    landmarks = []
    message = "Succes Face Detection"
    # try:
    imgData = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(imgData))
    image = image.convert('RGB')

    tm = cv2.TickMeter()
    tm.start()

    with torch.no_grad():
        _, landmarks = detect_faces(image, pnet, onet, rnet)

    tm.stop()
    logging.info(f"MTCNN face detection time: {tm.getTimeMilli()} miliseconds")
    tm.reset()
    # except Exception as err:
    #     message = "002"
    #     logging.info(f"face_alignment(): {err}")
    if (len(landmarks) == 0):
        message = "003"
    else:
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        tm.start()
        warped_face = warp_and_crop_face(
            np.array(image),
            facial5points,
            reference,
            crop_size=(crop_size, crop_size),
        )
        img_warp = Image.fromarray(warped_face)
        # img_warp.save("3.jpg")
        buffered = io.BytesIO()
        img_warp.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue())

    return img_warp, img_str, message