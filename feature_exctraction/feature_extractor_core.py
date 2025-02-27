import sys
import os
import logging
import torch
import torchvision.transforms as transforms
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

from face_detection.face_detector_core import face_alignment, mtcnn_detector, get_reference, base64, Image, io, cv2

import infranlib.databases.redis_handler as IMQ
import infranlib.databases.pool_handler as WKP

from embeddings import get_embedding
from backbone import Backbone

device = None
backbone = None
device = None
backbone = None
transform = None
input_size=[112, 112]

def init_model(gpu, model_path="checkpoint/backbone_ir50_ms1m_epoch120.pth"):

    global device
    global backbone
    global transform
    global input_size

    if(gpu==1):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            message_gpu = 'Using GPU'
        else:
            message_gpu = 'Using CPU'
    else:
        device = torch.device("cpu")
        message_gpu = 'Using CPU'

    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(model_path, map_location=device))
    backbone.to(device)
    backbone.eval()

    transform = transforms.Compose(
        [
            transforms.Resize([int(128 * input_size[0] / input_size[1]), int(128 * input_size[0] / input_size[1])]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )
    logging.info(f'[*] init_model(): {message_gpu} Done')
    

def feature_extraction(image_data, face_detector=True, face_alignment=face_alignment, logging=logging):
    global device
    global backbone
    global input_size
    global transform
    Embedding_d = np.zeros([1, 512])
    try:
        if face_detector:
            img_warp, img_str, err_code = face_alignment(image_data)
        else:
            imgData = base64.b64decode(image_data)
            img_warp = Image.open(io.BytesIO(imgData))
        
        tm = cv2.TickMeter()
        tm.start()
        embedding = get_embedding(backbone, img_warp, input_size, transform, device)
        #Embedding_d = np.zeros([1, 512])
        Embedding_d[0,:] = embedding
        tm.stop()
        logging.info(f"Feature Extraction time: {tm.getTimeMilli()} miliseconds")
        tm.reset()
        message = "Succes Feature Extraction"
        err_code = 0
    
    except Exception as err:
        logging.error(err)
                
        message = "Error Feature Extraction Model"
        err_code = 1

    return Embedding_d.dumps(), message, err_code