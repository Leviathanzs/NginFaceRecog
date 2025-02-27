import logging
import time
import msgpack
import base64
import datetime
import os
from inspect import getsourcefile
from os.path import abspath
import hashlib

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import face_detector_core
from read_config import readConfig

data_facedetect = readConfig()
HomeDir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
logging.getLogger().setLevel(logging.INFO)
# logging.basicConfig(filename=f'{HomeDir}/infran-detector-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.log',
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     force=True,
#                     level=logging.INFO)
# logger = logging.getLogger('DetectorFaceID')

hash_image = "aHR0cHM6Ly9pLmltZ3VyLmNvbS9jZG4xZ2J4LnBuZw=="
hash_object = hashlib.sha256()
image_str = ""

def detect_face_mtcnn(request):
    global hash_image
    global hash_object
    global image_str
    logging.info("Detect Face Using MTCNN is called.")
    message = "Success Face Detection"
    err_code = 0
    if(request['ImgData']==bytes('','utf-8')):
        message = "Error Request: imgData is Undefined"
        err_code = 1
    else:
        startt = time.time() * 1000
        img_str = ''
        if (request['ImgData'] == hash_image):
            logging.info("Hash Image is same")
            img_str = image_str
        else:
            hash_image = request['ImgData']
            logging.info("Hash Image is different")
        if (img_str == ''):
            try:

                _, img_str, err_code = face_detector_core.face_alignment(request['ImgData'])
                message = 'Success Face Detection'
                image_str = img_str

            except Exception as err:
                logging.error(err)
                
                message = "Error Face Detection Model"
                err_code = 1

        endt = time.time() * 1000
        duration = round(endt - startt)
        logging.info(f"Face Detection time server: {duration} milliseconds")

    data = {
        'TrxID': request['TrxID'],
        'Message': message, 
        'Result':'Get response from face detector', 
        'ErrCode':err_code,
        'Status': 'Get response from face detector',
        'ImgWarp': img_str,
        'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    }

    return data

class FaceDetectInfran:
    def __init__(self, channel, message, redis_conn, exec_name):
        logging.info("FaceDetectInfran is initialized from channel %s" % channel)
        self.exec_name = exec_name
        self.redis_conn = redis_conn
        if (message['Method'] == 'MTCNN'):
            self.DetectFaceMTCNN(message)
        # elif (message['method'] == 'VerifyFace'):
        #     self.VerifyFace(message)
        # elif (message['method'] == 'RegisterUser'):
        #     self.RegisterUser(message)
        # elif (message['method'] == 'RegisterUserMember'):
        #     self.RegisterUserMember(message)
        else:
            self.NotFound(message)
            
    def DetectFaceMTCNN(self, request, context=None):
        data = detect_face_mtcnn(request)

        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

    def NotFound(self, request, context=None):
        logging.info(f"Method not found {request['Method']}")
        data = {
            'TrxID': request['TrxID'],
            'Message': 'Method not found', 
            'Result':'Get response from face detector', 
            'ErrCode':1,
            'Status': 'Get response from face detector',
            'ImgWarp': '',
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }
        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

def serve(face_detector_id, exec_name):
    
    r_req_resp = face_detector_core.IMQ.RedisHandler(db=0)
    r_facedetect_pool = face_detector_core.IMQ.RedisHandler(db=3)

    start = time.time()
    face_detector_core.mtcnn_detector()
    face_detector_core.get_reference()
    end = time.time()
    logging.info(f"{exec_name} - [*] Load Face Detector Model: Done in {end-start} seconds")

    client = face_detector_core.IMQ.MessageListener(r_req_resp.get_connection(), exec_name, face_detector_id, FaceDetectInfran, logging)
    client.start()
    logging.info(f'{exec_name} - Message Listener is on')
    # r_command.set('worker_pool-xxxx', 1)
    client2 = face_detector_core.WKP.PoolNotifier(r_facedetect_pool, face_detector_id, prefix='facedetect_pool-')
    client2.start()
    logging.info(f'{exec_name} - Face Detect Pool Notifier is on')


if __name__ == "__main__":
    
    serve(data_facedetect['face_detector_id'], data_facedetect['face_detector_name'])