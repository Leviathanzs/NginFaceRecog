import logging
import time
import msgpack
import numpy as np
import datetime
import os
from inspect import getsourcefile
import uuid

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import feature_extractor_core
from read_config import readConfig

HomeDir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
logging.getLogger().setLevel(logging.INFO)
# logging.basicConfig(filename=f'{HomeDir}/infran-extractor-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.log',
#                     filemode='a',
#                     format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
#                     datefmt='%H:%M:%S',
#                     force=True,
#                     level=logging.INFO)
# logger = logging.getLogger('DetectorFaceID')

data_feature_extract = readConfig()

def feature_extractor(request):
    logging.info("Feature Extractor is called.")

    if(request['ImgData']==bytes('','utf-8')):
        message = "Error Request: imgData is Undefined"
        err_code = 1
    else:
        if('FaceDetector' in request.keys()):
            face_detector = True if request['FaceDetector']==0 else False
        else:
            face_detector = False
        # print(face_detector)
        startt = time.time() * 1000
        embedding = None
        try:

            embedding, message, err_code = feature_extractor_core.feature_extraction(request['ImgData'], face_detector, logging)

        except Exception as err:
            logging.error(err)
            
            message = "Error Feature Extraction Model"
            err_code = 1

        endt = time.time() * 1000
        duration = round(endt - startt)
        logging.info(f"All Service Feature Extractor time server: {duration} milliseconds")

    data = {
        'TrxID': request['TrxID'],
        'Message': message, 
        'Result':'Get response from feature extractor', 
        'ErrCode':err_code,
        'Status': 'Get response from feature extractor',
        'Embedding': embedding,
        'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    }

    return data

class FeatureExtractionInfran:
    def __init__(self, channel, message, redis_conn, exec_name):
        logging.info("FeatureExtractionInfran is initialized from channel %s" % channel)
        self.exec_name = exec_name
        self.redis_conn = redis_conn
        if (message['Method'] == 'feature'):
            self.FeatureExtraction(message)
        # elif (message['method'] == 'VerifyFace'):
        #     self.VerifyFace(message)
        # elif (message['method'] == 'RegisterUser'):
        #     self.RegisterUser(message)
        # elif (message['method'] == 'RegisterUserMember'):
        #     self.RegisterUserMember(message)
        else:
            self.NotFound(message)
            
    def FeatureExtraction(self, request, context=None):

        data = feature_extractor(request)

        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

    def NotFound(self, request, context=None):
        logging.info(f"Method not found {request['Method']}")
        data = {
            'TrxID': request['TrxID'],
            'Message': 'Method not found', 
            'Result':'Get response from feature extractor', 
            'ErrCode':1,
            'Status': 'Get response from feature extractor',
            'Embedding': '',
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }
        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

def serve(executor_id, exec_name):
    
    r_req_resp = feature_extractor_core.IMQ.RedisHandler(db=0)
    r_facedetect_pool = feature_extractor_core.IMQ.RedisHandler(db=3)

    start = time.time()
    feature_extractor_core.mtcnn_detector()
    feature_extractor_core.get_reference()
    feature_extractor_core.init_model(1)
    end = time.time()
    logging.info(f"{exec_name} - [*] Load Model: Done in {end-start} seconds")

    client = feature_extractor_core.IMQ.MessageListener(r_req_resp.get_connection(), exec_name, executor_id, FeatureExtractionInfran, logging)
    client.start()
    logging.info(f'{exec_name} - Message Listener is on')
    # r_command.set('worker_pool-xxxx', 1)
    client2 = feature_extractor_core.WKP.PoolNotifier(r_facedetect_pool, executor_id, prefix='featureextract_pool-')
    client2.start()
    logging.info(f'{exec_name} - Feature Extract Pool Notifier is on')


if __name__ == "__main__":
    
    serve(str(uuid.uuid4()), str(uuid.uuid4()))