
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

import infranlib.databases.redis_handler as IMQ
from face_detection.face_detector_service import detect_face_mtcnn, face_detector_core
from feature_exctraction.feature_extractor_service import feature_extractor, feature_extractor_core
import grpc
from grpcproto import executor_pb2_grpc
from grpcproto import executor_pb2
import struct
import tracemalloc
import time

start = time.time()
face_detector_core.mtcnn_detector()
face_detector_core.get_reference()
end = time.time()
print(f"Driver - [*] Load Face Detector Model: Done in {end-start} seconds")

start = time.time()
feature_extractor_core.mtcnn_detector()
feature_extractor_core.get_reference()
feature_extractor_core.init_model(1, os.path.dirname(os.path.abspath(__file__))+'/../feature_exctraction/checkpoint/backbone_ir50_ms1m_epoch120.pth')
end = time.time()
print(f"Driver - [*] Load Model: Done in {end-start} seconds")

class Executor():

    def __init__(self, executor_url='localhost:50051'):
        self.driver = executor_pb2_grpc.ExecutorInfranFaceIDStub(grpc.insecure_channel(executor_url))
    
    def IdentifyOne(self, TrxID, TenantID, EmbeddingID, EmbeddingFeature, Timestamp, 
                    DeviceID, EmbeddingBytes, EmbeddingString):
        reqs = executor_pb2.IdentifyOneRequest(TrxID=TrxID, TenantID=TenantID,
                                        EmbeddingID=EmbeddingID, EmbeddingFeature=EmbeddingFeature,
                                        Timestamp=Timestamp, DeviceID=DeviceID, EmbeddingBytes=EmbeddingBytes,
                                        EmbeddingString=EmbeddingString)
        return self.driver.IdentifyOne(reqs)
    
def IdentifyOne(executor_url, TrxID, TenantID, EmbeddingID, EmbeddingFeature, Timestamp, 
                DeviceID, EmbeddingBytes, EmbeddingString):
    driver =executor_pb2_grpc.ExecutorInfranFaceIDStub(grpc.insecure_channel(executor_url))
    reqs = executor_pb2.IdentifyOneRequest(TrxID=TrxID, TenantID=TenantID,
                                    EmbeddingID=EmbeddingID, EmbeddingFeature=EmbeddingFeature,
                                    Timestamp=Timestamp, DeviceID=DeviceID, EmbeddingBytes=EmbeddingBytes,
                                    EmbeddingString=EmbeddingString)
    return driver.IdentifyOne(reqs)

def IdentifyOneAsync(IMQ: IMQ, r_req_resp : IMQ.RedisHandler, logging, executor_id, TrxID, TenantID, EmbeddingID, EmbeddingFeature, Timestamp, 
                DeviceID, EmbeddingBytes, EmbeddingString):
    # tracemalloc.start()  # Start tracing memory allocations
    ReturnChannel = f"{executor_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Executor', 
        'Result':'Not Get response from executor', 
        'ErrCode': 1,
        'Status': 'Not Get response from executor',
        'EmbeddingID': EmbeddingID,
        'MatchEmbeddingID': 'id_embedding',
        'ConfidenceScore': 0.0,
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'TenantID': TenantID,
            'EmbeddingID': EmbeddingID,
            'EmbeddingFeature': EmbeddingFeature,
            'EmbeddingBytes': EmbeddingBytes,
            'EmbeddingString': EmbeddingString,
            'Timestamp': Timestamp,
            'DeviceID': DeviceID,
            'Method': 'IdentifyFace'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), executor_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), executor_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()
        

        # tracemalloc.stop()  # Stop tracing memory allocations
    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Executor Error'
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"{ReturnChannel} => Current memory usage: {current / (1024 ** 2)} MB")
    # print(f"{ReturnChannel} => Peak memory usage: {peak / (1024 ** 2)} MB")
    return response

async def IdentifyOneAsyncio(IMQ, r_req_resp, logging, executor_id, TrxID, TenantID, EmbeddingID, EmbeddingFeature, Timestamp, 
                DeviceID, EmbeddingBytes, EmbeddingString):
    # tracemalloc.start()  # Start tracing memory allocations
    ReturnChannel = f"{executor_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Executor', 
        'Result':'Not Get response from executor', 
        'ErrCode': 1,
        'Status': 'Not Get response from executor',
        'EmbeddingID': EmbeddingID,
        'MatchEmbeddingID': 'id_embedding',
        'ConfidenceScore': 0.0,
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'TenantID': TenantID,
            'EmbeddingID': EmbeddingID,
            'EmbeddingFeature': EmbeddingFeature,
            'EmbeddingBytes': EmbeddingBytes,
            'EmbeddingString': EmbeddingString,
            'Timestamp': Timestamp,
            'DeviceID': DeviceID,
            'Method': 'IdentifyFace'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), executor_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), executor_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()
        

        # tracemalloc.stop()  # Stop tracing memory allocations
    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Executor Error'
    # current, peak = tracemalloc.get_traced_memory()
    # print(f"{ReturnChannel} => Current memory usage: {current / (1024 ** 2)} MB")
    # print(f"{ReturnChannel} => Peak memory usage: {peak / (1024 ** 2)} MB")
    return response

def IdentifyManyAsync(IMQ, r_req_resp, logging, executor_id, TrxID, TenantID, EmbeddingID, EmbeddingFeature, Timestamp, 
                DeviceID, EmbeddingBytes, EmbeddingString, NumResult):
    ReturnChannel = f"{executor_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Executor', 
        'Result':'Not Get response from executor', 
        'ErrCode': 1,
        'Status': 'Not Get response from executor',
        'EmbeddingID': EmbeddingID,
        'MatchEmbeddingID': 'id_embedding',
        'ConfidenceScore': 0.0,
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'TenantID': TenantID,
            'EmbeddingID': EmbeddingID,
            'EmbeddingFeature': EmbeddingFeature,
            'EmbeddingBytes': EmbeddingBytes,
            'EmbeddingString': EmbeddingString,
            'Timestamp': Timestamp,
            'DeviceID': DeviceID,
            'NumResult': NumResult,
            'Method': 'IdentifyManyFace'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), executor_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), executor_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Executor Error'

    return response

def RegisterUserAsync(IMQ, r_req_resp, logging, executor_id, TrxID, TenantID, EmbeddingID, Timestamp, DeviceID):
    ReturnChannel = f"{executor_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Executor', 
        'Result':'Not Get response from executor', 
        'EmbeddingID': EmbeddingID,
        'ErrCode': 1,
        'Status': 'Not Get response from executor',
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'TenantID': TenantID,
            'EmbeddingID': EmbeddingID,
            'Timestamp': Timestamp,
            'DeviceID': DeviceID,
            'Method': 'RegisterFace'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), executor_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), executor_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Executor Error'

    return response

def DeleteUserAsync(IMQ, r_req_resp, logging, executor_id, TrxID, TenantID, EmbeddingIDs, Timestamp, DeviceID):
    ReturnChannel = f"{executor_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Executor', 
        'Result':'Not Get response from executor', 
        'EmbeddingIDs': EmbeddingIDs,
        'ErrCode': 1,
        'Status': 'Not Get response from executor',
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'TenantID': TenantID,
            'EmbeddingIDs': EmbeddingIDs,
            'Timestamp': Timestamp,
            'DeviceID': DeviceID,
            'Method': 'DeleteFace'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), executor_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), executor_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()
        

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Executor Error'

    return response

def FaceDetectorAsync(IMQ, r_req_resp, logging, face_detector_id, ImgData, TrxID, Timestamp):
    ReturnChannel = f"{face_detector_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Face Detector', 
        'Result':'Not Get response from face detector', 
        'ErrCode':1,
        'Status': 'Not Get response from face detector',
        'ImgWarp': '',
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'ImgData': ImgData,
            'Method': 'MTCNN'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), face_detector_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), face_detector_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Face Detector Error'

    return response
  
def FaceDetectorFunc(ImgData, TrxID, Timestamp):
    # ReturnChannel = f"{face_detector_id}-{TrxID}"
    response = {
        'TrxID': TrxID,
        'Message': 'Error Face Detector', 
        'Result':'Not Get response from face detector', 
        'ErrCode':1,
        'Status': 'Not Get response from face detector',
        'ImgWarp': '',
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            # 'ReturnChannel': ReturnChannel,
            'ImgData': ImgData,
            'Method': 'MTCNN'
        }
        # print('ke message publish')
        # messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), face_detector_id, req, logging)
        # # print('ke message listener')
        # client = IMQ.MessageListener(r_req_resp.get_connection(), face_detector_id, ReturnChannel, None, logging, send_message=messagep)
        # client.start()
        # print('ke response')
        # response = client.WaitForResult()

        response = detect_face_mtcnn(req)

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Face Detector Error'

    return response

def FeatureExtractorAsync(IMQ, r_req_resp, logging, feature_extractor_id, ImgData, TrxID, Timestamp):
    ReturnChannel = f"{feature_extractor_id}-{TrxID}"
    if not r_req_resp.check_connection():
        rconn = r_req_resp.get_connection()
    response = {
        'TrxID': TrxID,
        'Message': 'Error Feature Extractor', 
        'Result':'Not Get response from face detector', 
        'ErrCode':1,
        'Status': 'Not Get response from face detector',
        'Embedding': [],
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            'ReturnChannel': ReturnChannel,
            'ImgData': ImgData,
            'FaceDetector':1,
            'Method': 'feature'
        }
        # print('ke message publish')
        messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), feature_extractor_id, req, logging)
        # print('ke message listener')
        client = IMQ.MessageListener(r_req_resp.get_connection(), feature_extractor_id, ReturnChannel, None, logging, send_message=messagep)
        client.start()
        # print('ke response')
        response = client.WaitForResult()

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Feature Extractor Error'

    return response

def FeatureExtractorFunc(ImgData, TrxID, Timestamp):
    # ReturnChannel = f"{feature_extractor_id}-{TrxID}"
    response = {
        'TrxID': TrxID,
        'Message': 'Error Feature Extractor', 
        'Result':'Not Get response from face detector', 
        'ErrCode':1,
        'Status': 'Not Get response from face detector',
        'Embedding': [],
        'Timestamp': Timestamp
    }
    try:
        # print('ke json')
        req = {
            'TrxID': TrxID,
            # 'ReturnChannel': ReturnChannel,
            'ImgData': ImgData,
            'FaceDetector':1,
            'Method': 'feature'
        }
        # # print('ke message publish')
        # messagep = IMQ.MessagePublisher(r_req_resp.get_connection(), feature_extractor_id, req, logging)
        # # print('ke message listener')
        # client = IMQ.MessageListener(r_req_resp.get_connection(), feature_extractor_id, ReturnChannel, None, logging, send_message=messagep)
        # client.start()
        # # print('ke response')
        # response = client.WaitForResult()
        response = feature_extractor(req)

    except Exception as err:
        print(f'error : {err}')
        response['Message'] = 'Feature Extractor Error'

    return response