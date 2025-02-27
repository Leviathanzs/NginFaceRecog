import base64
from os import path
import time
import struct
from PIL import Image
import io
import json
from collections import Counter
import pandas as pd
import random
import itertools

from concurrent import futures
import logging
import uuid
import math
import numpy as np
import gc

import grpc
from grpcproto import infrandriver_pb2_grpc
from grpcproto import infrandriver_pb2

import sys
import os
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv(os.path.dirname(os.path.abspath(__file__))+'/.env')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

import infranlib.databases.redis_handler as IMQ
import infranlib.databases.pool_handler as WKP

from inspect import getsourcefile
from os.path import abspath
from datetime import datetime

HomeDir = path.dirname(path.abspath(getsourcefile(lambda:0)))
logging.getLogger().setLevel(logging.INFO)
logging.info(f"HomeDir: {HomeDir}")

logging.basicConfig(filename=f'{HomeDir}/infran-controller-{datetime.now().strftime("%Y%m%dT%H%M%S")}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    force=True,
                    level=logging.INFO)
logger = logging.getLogger('ControllerFaceID')

write_sequence = False
last_time_write_sequence =  time.time()

r_worker_pool = IMQ.RedisHandler(db=3)
reqs_count = 0

inactive_driver = ['10.10.101.4:9099']

class DriverInfranFaceID(infrandriver_pb2_grpc.DriverInfranFaceIDServicer):
    def VerifyById (self, request, context):
        global write_sequence
        global reqs_count
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get VerifyById Request")

        TrxID = request.TrxID
        ImgData = request.ImgData
        UserID = request.UserID

        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())

        while write_sequence:
            time.sleep(0.1)
            if time.time() - last_time_write_sequence > 10:
                break

        poolget = WKP.PoolGetter(r_worker_pool)
        driver_ids = poolget.GetAllWorker('driver_pool-')
        driver_ids = [channel.split('-')[-1] for channel in driver_ids]
        logging.info(f"Available Driver IDs: {driver_ids}")
        
        driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
        if not driver_ids:
            logging.error("No active drivers available!")
            return infrandriver_pb2.VerifyResponse(
                TrxID=TrxID,
                ErrCode="1",
                Message="No active drivers",
                Timestamp=str(datetime.now()),
                ConfidenceScore=0.0
            )

        driver_cycle = itertools.cycle(driver_ids)
        driver_id = next(driver_cycle)
        logging.info(f"Selected Driver ID: {driver_id}")

        try:
            driver = infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(driver_id))
            reqs = infrandriver_pb2.VerifyByIdRequest(TrxID=TrxID, ImgData=ImgData, UserID=UserID)
            logging.info(f"Sending request to Driver: {driver_id}")
            resp = driver.VerifyById(reqs)
        except grpc.RpcError as e:
            logging.error(f"GRPC Error: {e.code()} - {e.details()}")
            return infrandriver_pb2.VerifyResponse(
                TrxID=TrxID,
                ErrCode="2",
                Message=f"GRPC Error: {e.code()}",
                Timestamp=str(datetime.now()),
                ConfidenceScore=0.0
            )

        return resp

    def VerifyByImage (self, request, context):
        global write_sequence
        global reqs_count
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get VerifyByImage Request")

        TrxID = request.TrxID
        ImgData1 = request.ImgData1
        ImgData2 = request.ImgData2

        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        
        while write_sequence:
            time.sleep(0.1)
            if time.time() - last_time_write_sequence > 10:
                break

        poolget = WKP.PoolGetter(r_worker_pool)
        driver_ids = poolget.GetAllWorker('driver_pool-')
        driver_ids = [channel.split('-')[-1] for channel in driver_ids]
        logging.info(f"Available Driver IDs: {driver_ids}")

        driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
        if not driver_ids:
            logging.error("No active drivers available!")
            return infrandriver_pb2.VerifyResponse(
                TrxID=TrxID,
                ErrCode="1",
                Message="No active drivers",
                Timestamp=str(datetime.now()),
                ConfidenceScore=0.0
            )

        driver_cycle = itertools.cycle(driver_ids)
        driver_id = next(driver_cycle)
        logging.info(f"Selected Driver ID: {driver_id}")

        try:
            driver = infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(driver_id))
            reqs = infrandriver_pb2.VerifyByImageRequest(TrxID=TrxID, ImgData1=ImgData1, ImgData2=ImgData2)
            logging.info(f"Sending request to Driver: {driver_id}")
            resp = driver.VerifyByImage(reqs)
        except grpc.RpcError as e:
            logging.error(f"GRPC Error: {e.code()} - {e.details()}")
            return infrandriver_pb2.VerifyResponse(
                TrxID=TrxID,
                ErrCode="2",
                Message=f"GRPC Error: {e.code()}",
                Timestamp=str(datetime.now()),
                ConfidenceScore=0.0
            )

        return resp

    def IdentifyOne(self, request, context):
        global write_sequence
        global reqs_count
        fields = ['Waktu Face Detect', 'Waktu Feature Extract', 'Waktu Collections', 'Memory Usage']
        filename = "./Result_Driver_thread(tessar).csv"
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get IdentifyOne Request")
        TrxID = request.TrxID
        ImgData = request.ImgData
        FaceDetector = request.FaceDetector
        EmbeddingFeature = list(request.EmbeddingFeature)
        Result = "-1"
        Status = "Failed"
        ErrCode = "999"
        Timestamp = request.Timestamp
        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        if (request.Timestamp==''):
            Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        if(request.TenantID==''):
            ErrCode = "011"
        while write_sequence:
            time.sleep(0.1)
            if time.time() - last_time_write_sequence > 10:
                break
        
        poolget = WKP.PoolGetter(r_worker_pool)
        driver_ids = poolget.GetAllWorker('driver_pool-')
        driver_ids = [channel.split('-')[-1] for channel in driver_ids]
        logging.info(f"Driver IDs: {driver_ids}")
        driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
        driver_cycle = itertools.cycle(driver_ids)
        driver_id = next(driver_cycle)
        logging.info(f"Driver ID: {driver_id}")
        
        try:
            driver =infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(driver_id))
            reqs = infrandriver_pb2.IdentifyOneRequest(TrxID=TrxID, TenantID=request.TenantID,
                                        Timestamp=Timestamp, DeviceID=request.DeviceID, 
                                        ImgData=ImgData, FaceDetector=FaceDetector, EmbeddingFeature=EmbeddingFeature)
            resp = driver.IdentifyOne(reqs)
        except grpc.RpcError as e:
            logging.error(f"Status Code GRPC: {e.code()}")
            resp = driver.IdentifyOne(reqs)
        return resp

    def IdentifyMany(self, request, context):
        global write_sequence
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get IdentifyMany Request")
        TrxID = request.TrxID
        ImgData = request.ImgData
        FaceDetector = request.FaceDetector
        Result = "-1"
        Status = "Failed"
        ErrCode = "999"
        Timestamp = request.Timestamp
        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        if (request.Timestamp==''):
            Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        if(request.TenantID==''):
            ErrCode = "011"
        while write_sequence:
            time.sleep(0.1)
            if time.time() - last_time_write_sequence > 10:
                break

        poolget = WKP.PoolGetter(r_worker_pool)
        driver_ids = poolget.GetAllWorker('driver_pool-')
        driver_ids = [channel.split('-')[-1] for channel in driver_ids]
        logging.info(f"Driver IDs: {driver_ids}")
        driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
        driver_cycle = itertools.cycle(driver_ids)
        driver_id = next(driver_cycle)
        logging.info(f"Driver ID: {driver_id}")
        try:
            driver =infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(driver_id))
            reqs = infrandriver_pb2.IdentifyManyRequest(TrxID=TrxID, TenantID=request.TenantID,
                                        Timestamp=Timestamp, DeviceID=request.DeviceID, 
                                        ImgData=ImgData, FaceDetector=FaceDetector, NumResult=request.NumResult)
            resp = driver.IdentifyMany(reqs)
        except grpc.RpcError as e:
            logging.error("Status Code GRPC: "+e.code())
            resp = driver.IdentifyMany(reqs)
        return resp

    def RegisterUser(self, request, context):
        global write_sequence
        global last_time_write_sequence
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get RegisterUser Request")
        TrxID = request.TrxID
        ImgData = request.ImgData
        UserID = request.UserID
        Result = "-1"
        Status = "Failed"
        ErrCode = "999"
        Message = "Internal Server Error"
        Timestamp = request.Timestamp
        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        if (request.Timestamp==''):
            Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        if(request.TenantID==''):
            ErrCode = "011"

        write_sequence = True
        last_time_write_sequence =  time.time()

        poolget = WKP.PoolGetter(r_worker_pool)
        driver_ids = poolget.GetAllWorker('driver_pool-')
        driver_ids = [channel.split('-')[-1] for channel in driver_ids]
        logging.info(f"Driver ID: {driver_ids}")
        driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
        driver_cycle = itertools.cycle(driver_ids)
        driver_id = next(driver_cycle)
        logging.info(f"Driver ID: {driver_id}")
        TenantID = request.TenantID
        DeviceID = request.DeviceID
        Name = request.Name
        Description = request.Description
        UserID = request.UserID
        abortEnrollment = False  
        register_db_driver_id = driver_id
        driver =infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(register_db_driver_id))
        reqs = infrandriver_pb2.RegisterUserRequest(TrxID=TrxID, TenantID=request.TenantID,
                                Timestamp=Timestamp, DeviceID=DeviceID, 
                                ImgData=ImgData, FaceDetector=request.FaceDetector, 
                                Name=request.Name, Description=request.Description, UserID=request.UserID)
        logging.info(f"Request data: {reqs}")
        resp = driver.RegisterUser(reqs)
        if resp.ErrCode != "0":
            if resp.Message != "Face Duplicate":
                resp_delete = self.DeleteUser(infrandriver_pb2.DeleteUserRequest(TrxID=TrxID, TenantID=TenantID,
                                                Timestamp=Timestamp, DeviceID=DeviceID,
                                                UserID=resp.UserID), context)
        else:
            driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
            rkey = "{0}#{1}#{2}".format(request.TenantID, resp.UserID, resp.UserUniqueId)
            for channel in driver_ids:
                driver =infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(channel))
                reqs = infrandriver_pb2.RegisterUserRequest(TrxID=TrxID, TenantID=request.TenantID,
                                Timestamp=Timestamp, DeviceID=DeviceID, 
                                ImgData=ImgData, FaceDetector=request.FaceDetector, 
                                Name=request.Name, Description=request.Description, UserID=request.UserID)
                resp_ = driver.RegisterUser(reqs)
        write_sequence = False
        return resp

    def DeleteUser(self, request, context):
        global write_sequence
        global last_time_write_sequence
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get DeleteUser Request")
        TrxID = request.TrxID
        UserID = request.UserID
        Result = "-1"
        Status = "Failed"
        ErrCode = "999"
        Timestamp = request.Timestamp
        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        if (request.Timestamp==''):
            Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        if(request.TenantID==''):
            ErrCode = "011"
        
        write_sequence = True
        last_time_write_sequence =  time.time()

        TenantID = request.TenantID
        DeviceID = request.DeviceID

        poolget = WKP.PoolGetter(r_worker_pool)
        driver_ids = poolget.GetAllWorker('driver_pool-')
        driver_ids = [channel.split('-')[-1] for channel in driver_ids]
        logging.info(f"Driver ID: {driver_ids}")
        driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
        driver_cycle = itertools.cycle(driver_ids)
        driver_id = next(driver_cycle)
        logging.info(f"Driver ID: {driver_id}")

        delete_db_driver_id = driver_id
        driver =infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(delete_db_driver_id))
        reqs = infrandriver_pb2.DeleteUserRequest(TrxID=TrxID, TenantID=TenantID,
                            Timestamp=Timestamp, DeviceID=DeviceID,
                            UserID=UserID)
        resp = driver.DeleteUser(reqs)

        if resp.ErrCode == "0":
            driver_ids = [ip for ip in driver_ids if ip not in inactive_driver]
            rkey = resp.rkey
            for channel in driver_ids:
                driver =infrandriver_pb2_grpc.DriverInfranFaceIDStub(grpc.insecure_channel(channel))
                reqs = infrandriver_pb2.DeleteUserRequest(TrxID=TrxID, TenantID=TenantID,
                                    Timestamp=Timestamp, DeviceID=DeviceID,
                                    UserID=UserID, rkey=rkey)
                resp = driver.DeleteUser(reqs)
        write_sequence = False

        return resp

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    infrandriver_pb2_grpc.add_DriverInfranFaceIDServicer_to_server(DriverInfranFaceID(), server)
    server.add_insecure_port('[::]:%s'%(os.getenv('PORT')))
    server.start()
    try:  
        server.wait_for_termination()
    except Exception as e: 
        print(e)

if __name__ == '__main__':
    base64_string = base64.b64encode("GeeksForGeeks is the best".encode()).decode()
    reconverted = base64.b64decode(base64_string).decode()
    
    serve()
    
