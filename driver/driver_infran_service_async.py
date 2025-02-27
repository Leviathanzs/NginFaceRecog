import base64
from os import path
import time
import struct
from PIL import Image
import io
import json
from collections import Counter
import pandas as pd

from concurrent import futures
import logging
import uuid
import math
import numpy as np
import psutil
import gc
import ast
import asyncutils as AU

import grpc
from grpcproto import infrandriver_pb2_grpc
from grpcproto import infrandriver_pb2

import sys
import os
from datetime import datetime
import time
from dotenv import load_dotenv

load_dotenv(os.path.dirname(os.path.abspath(__file__))+'/.env')

import executor as EXE

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/..')

import infranlib.databases.redis_handler as IMQ
import infranlib.databases.pool_handler as WKP
import infranlib.databases.postgres_handler as DBP
import infranlib.databases.database_query as DBQ

import concurrent.futures
from inspect import getsourcefile
from os.path import abspath
from datetime import datetime
import csv
import torch.nn.functional as F
import pickle


HomeDir = path.dirname(path.abspath(getsourcefile(lambda:0)))
logging.getLogger().setLevel(logging.INFO)
logging.info(f"HomeDir: {HomeDir}")

logging.basicConfig(filename=f'{HomeDir}/infran-driver-{datetime.now().strftime("%Y%m%dT%H%M%S")}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    force=True,
                    level=logging.INFO)
logger = logging.getLogger('DriverFaceID')

r_req_resp = IMQ.RedisHandler(db=0)
r_worker_pool = IMQ.RedisHandler(db=3)
r_proc_feature = IMQ.RedisHandler(db=1)
dbh = DBP.DatabaseHandler()
reqs_count = 0
local_host_name = os.getenv('HOST')

def cosine_similarity(emb1, emb2):
    # Ensure embeddings are 1D arrays
    emb1 = np.array(emb1).flatten()
    emb2 = np.array(emb2).flatten()

    # Compute dot product and norms
    dot_product = np.dot(emb1, emb2)
    norm_emb1 = np.linalg.norm(emb1)
    norm_emb2 = np.linalg.norm(emb2)
    
    # Compute cosine similarity
    cos_sim = dot_product / (norm_emb1 * norm_emb2)
    
    # Clip the result to handle possible floating point issues
    cos_sim = np.clip(cos_sim, -1.0, 1.0)
    return cos_sim

class DriverInfranFaceID(infrandriver_pb2_grpc.DriverInfranFaceIDServicer):
    def VerifyById(self, request, context):
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f") + " Get verifyById Request")
        
        TrxID = request.TrxID
        ImgData = request.ImgData
        UserID = request.UserID

        # Validasi input
        if not TrxID:
            TrxID = str(uuid.uuid4())
        if not ImgData:
            return infrandriver_pb2.VerifyResponse(
                TrxID=TrxID, ErrCode="1", Message="ImgData is required", Timestamp=datetime.now().strftime("%Y%m%dT%H%M%S")
            )
        if not UserID:
            return infrandriver_pb2.VerifyResponse(
                TrxID=TrxID, ErrCode="1", Message="UId is required", Timestamp=datetime.now().strftime("%Y%m%dT%H%M%S")
            )

        # Deteksi wajah
        resp_face_detector1 = EXE.FaceDetectorFunc(ImgData, TrxID, datetime.now().strftime("%Y%m%dT%H%M%S"))
        ImgData1 = resp_face_detector1['ImgWarp']

        # Ekstraksi fitur wajah
        resp_feature_extractor1 = EXE.FeatureExtractorFunc(ImgData1, TrxID, datetime.now().strftime("%Y%m%dT%H%M%S"))
        Embedding1 = resp_feature_extractor1['Embedding']
        embedding1 = pickle.loads(Embedding1)

        # Mengambil embedding_data dari database
        embedding_data = DBQ.loadEmbeddings(dbh, UserID)
        totalIndex = len(embedding_data)
        embedding_data_value = embedding_data[0]
        embedding_array = embedding_data_value[-1]
        embedding_array = ast.literal_eval(embedding_array)
        embedding_array = np.array(embedding_array)
       
        if embedding_array.ndim == 1:
            embedding_array = embedding_array.reshape(1, -1)

        similarity = cosine_similarity(embedding1, embedding_array)
        score = similarity

        return infrandriver_pb2.VerifyResponse(
            TrxID=TrxID, ErrCode="0", Message="Sukses", Timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"),
            ConfidenceScore=score
        )

        
    def VerifyByImage(self, request, context):
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get verifyByImage Request")
        TrxID = request.TrxID
        ImgData1 = request.ImgData1
        ImgData2 = request.ImgData2

        if(TrxID == '' or TrxID == None):
            TrxID = str(uuid.uuid4())
        if(ImgData1 == '' or ImgData1 == None):
            return infrandriver_pb2.VerifyResponse(TrxID=TrxID, ErrCode="1",Message="ImgData1 is required",Timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"))
        if(ImgData2 == '' or ImgData2 == None):
            return infrandriver_pb2.VerifyResponse(TrxID=TrxID, ErrCode="1",Message="ImgData2 is required",Timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"))
        
        ## Inference
        resp_face_detector1 = EXE.FaceDetectorFunc(ImgData1, TrxID, datetime.now().strftime("%Y%m%dT%H%M%S"))
        ImgData1 = resp_face_detector1['ImgWarp']

        resp_face_detector2 = EXE.FaceDetectorFunc(ImgData2, TrxID, datetime.now().strftime("%Y%m%dT%H%M%S"))
        ImgData2 = resp_face_detector2['ImgWarp']

        resp_feature_extractor1 = EXE.FeatureExtractorFunc(ImgData1, TrxID, datetime.now().strftime("%Y%m%dT%H%M%S"))
        Embedding1 = resp_feature_extractor1['Embedding']

        resp_feature_extractor2 = EXE.FeatureExtractorFunc(ImgData2, TrxID, datetime.now().strftime("%Y%m%dT%H%M%S"))
        Embedding2 = resp_feature_extractor2['Embedding']
 
        embedding1 = pickle.loads(Embedding1)
        embedding2 = pickle.loads(Embedding2)

        # Calculate cosine similarity
        similarity = cosine_similarity(embedding1, embedding2)
        score = similarity
        response = infrandriver_pb2.VerifyResponse(TrxID=TrxID, ErrCode="0", Message="Sukses", Timestamp=datetime.now().strftime("%Y%m%dT%H%M%S"), ConfidenceScore=score)
        return response
   
    def IdentifyOne(self, request, context):
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
        result_facedetect = 0
        result_feature_extraction = 0
        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        if (request.Timestamp==''):
            Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        if(request.TenantID==''):
            ErrCode = "011"

        poolget = WKP.PoolGetter(r_worker_pool)
        if(len(request.EmbeddingFeature)==0):

            if(request.FaceDetector==0):
                ###-----------------Get Face Detector-----------------###
                start_facedetect = time.time()*1000
                resp_face_detector = EXE.FaceDetectorFunc(ImgData, TrxID, Timestamp)
                ImgData = resp_face_detector['ImgWarp']
                finish_facedetect = time.time()*1000
                logging.info(f'Face Detection time : {round(finish_facedetect-start_facedetect)} miliseconds')
                result_facedetect = round(finish_facedetect-start_facedetect)

            ###-----------------Get Feature Extraction-----------------###
            start_featureextract = time.time()*1000
            resp_feature_extractor = EXE.FeatureExtractorFunc(ImgData, TrxID, Timestamp)
            EmbeddingFeature = resp_feature_extractor['Embedding']
            EmbeddingFeature = np.load(io.BytesIO(EmbeddingFeature), allow_pickle=True)
            EmbeddingFeature = list(pd.DataFrame(EmbeddingFeature).values[0])
            finish_featureextract= time.time()*1000
            logging.info(f'Feature Extractor time : {round(finish_featureextract-start_featureextract)} miliseconds')
            result_feature_extraction = round(finish_featureextract - start_featureextract)
        ###-----------------Get Executor Destination-----------------###

        start_identification = time.time()*1000
        worker_id = poolget.GetAllWorker('executor_pool-%s-'%(local_host_name))
        executor_ids = []
        for i in worker_id:
            executor_ids.append(str(i).replace('executor_pool-%s-'%(local_host_name),''))
        logging.info(f"Executor ID: {executor_ids}")
        get_executor_checkpoint_time = time.time()*1000
        logging.info(f'Get All Executor time : {round(get_executor_checkpoint_time-start_identification)} miliseconds')

        ###-----------------Get Embedding Base List-----------------###
        EmbeddingString = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        RespPoll = []
        MaxWorker = 1
        if (len(executor_ids) > MaxWorker):
            MaxWorker = len(executor_ids)
        embedding_list_checkpoint_time = time.time()*1000
        ###-----------------------Save embedding feature to redis-----------------------###
        process = psutil.Process()
        memory_info = process.memory_info()
        logging.info(f"Memory Usage Before Thread-Pool : {memory_info.rss/1024/1024} MB")
        ###-------------------------------------------------------###
        with concurrent.futures.ThreadPoolExecutor(max_workers=MaxWorker) as executor:
            # Start the load operations and mark each future with its URL
            future_to_exec = {executor.submit(EXE.IdentifyOneAsync, IMQ, r_req_resp, logging, executor_id, TrxID=TrxID, 
                                            TenantID=request.TenantID, 
                                            EmbeddingID=TrxID, 
                                            EmbeddingFeature=EmbeddingFeature,
                                            EmbeddingBytes=bytes('','utf-8'),
                                            Timestamp=Timestamp, DeviceID=request.DeviceID, 
                                            EmbeddingString=estring): (executor_id, estring) for executor_id, estring in zip(executor_ids, EmbeddingString)}
            
            for future in concurrent.futures.as_completed(future_to_exec):
                memory_info = process.memory_info()
                logging.info(f"Memory Usage Thread-Pool : {memory_info.rss/1024/1024} MB")
                url = future_to_exec[future]
                try:
                    RespPoll.append(future.result())
                except Exception as ex:
                    logging.info('%r generated an exception: %s' % (url, ex))
        ###-----------------Send Request to Worker-----------------###
        
        max_val_list = [subResp['ConfidenceScore'] for subResp in RespPoll]
        max_val = max(max_val_list)
        max_idx = max_val_list.index(max_val)
        max_val_resp = RespPoll[max_idx]
        pool_executor_response_checkpoint_time = time.time()*1000
        logging.info(f'Concurent All Executor time : {round(pool_executor_response_checkpoint_time-embedding_list_checkpoint_time)} miliseconds')
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" IdentifyOne Request Finished\n")
        logging.info("------------------------------------------------------------------------------")
        logging.info("\n")
        collecting_time = round(pool_executor_response_checkpoint_time-embedding_list_checkpoint_time)
        memory_info = process.memory_info()
        logging.info(f"Memory Usage: {memory_info.rss/1024/1024} MB")
        rows = [result_facedetect, result_feature_extraction, collecting_time, memory_info.rss/1024/1024]
        if os.path.exists(filename):         
            with open(filename, 'a') as csvfile: 
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(rows)
        collecting_time = round(pool_executor_response_checkpoint_time-get_executor_checkpoint_time)
        if max_val_resp['ConfidenceScore'] < 1 and max_val_resp['ConfidenceScore'] > 0.8:
                        # Create the data structure to store embedding ID and the list
            data_to_save = {
                "embedding_id": max_val_resp['MatchEmbeddingID'],
                "embedding": EmbeddingFeature,
                "score": max_val_resp['ConfidenceScore']
            }

            # Save the embedding ID and list to a JSON file with new lines
            with open('embedding_list_with_id.json', 'a') as f:
                json.dump(data_to_save, f, indent=4)  # Writing with indentation for readability

            print(f"Embedding list with ID {max_val_resp['MatchEmbeddingID']} saved to file.")

        ###-----------------------Remove embedding feature to redis-----------------------###
        reqs_count += 1
        if (reqs_count > 1000):
            reqs_count = 0
            gc.collect()
        return infrandriver_pb2.IdentifyOneResponse(TrxID=TrxID, Result=max_val_resp['Result'],
                                                    Status=max_val_resp['Status'], ErrCode=str(max_val_resp['ErrCode']), EmbeddingID=max_val_resp['EmbeddingID'],
                                                    MatchEmbeddingID=max_val_resp['MatchEmbeddingID'], ConfidenceScore=max_val_resp['ConfidenceScore'],
                                                    Message=max_val_resp['Message'], Timestamp=Timestamp)

    def IdentifyMany(self, request, context):
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
        # elif(request.EmbeddingID==''):
        #     ErrCode = "012"
        # elif(request.EmbeddingFeature==bytes('','utf-8')):
        #     ErrCode = "013"

        poolget = WKP.PoolGetter(r_worker_pool)
        if(request.FaceDetector==0):
            ###-----------------Get Face Detector-----------------###
            start_facedetect = time.time()*1000
            # facedetector_id = poolget.GetWorker('facedetect_pool-')
            # logging.info(f"Face Detection ID: {facedetector_id}")
            # resp_face_detector = EXE.FaceDetectorAsync(IMQ, r_req_resp, logging, facedetector_id, ImgData, request.TrxID, Timestamp)
            resp_face_detector = EXE.FaceDetectorFunc(ImgData, TrxID, Timestamp)
            ImgData = resp_face_detector['ImgWarp']
            finish_facedetect = time.time()*1000
            logging.info(f'Face Detection time : {round(finish_facedetect-start_facedetect)} miliseconds')

        ###-----------------Get Feature Extraction-----------------###
        start_featureextract = time.time()*1000
        # featureextractor_id = poolget.GetWorker('featureextract_pool-')
        # logging.info(f"Feature Extractor ID: {featureextractor_id}")
        # resp_feature_extractor = EXE.FeatureExtractorAsync(IMQ, r_req_resp, logging, featureextractor_id, ImgData, request.TrxID, Timestamp)
        resp_feature_extractor = EXE.FeatureExtractorFunc(ImgData, TrxID, Timestamp)
        EmbeddingFeature = resp_feature_extractor['Embedding']
        EmbeddingFeature = np.load(io.BytesIO(EmbeddingFeature), allow_pickle=True)
        EmbeddingFeature = list(pd.DataFrame(EmbeddingFeature).values[0])
        finish_featureextract= time.time()*1000
        logging.info(f'Feature Extractor time : {round(finish_featureextract-start_featureextract)} miliseconds')
        # else:
        #     Embedding = list(request.EmbeddingFeature)
        ###-----------------Get Executor Destination-----------------###

        start_identification = time.time()*1000
        worker_id = poolget.GetAllWorker('executor_pool-%s-'%(local_host_name))
        executor_ids = []
        for i in worker_id:
            executor_ids.append(str(i).replace('executor_pool-%s-'%(local_host_name),''))
        logging.info(f"Executor ID: {executor_ids}")
        get_executor_checkpoint_time = time.time()*1000
        logging.info(f'Get All Executor time : {round(get_executor_checkpoint_time-start_identification)} miliseconds')
        ###-------------------------------------------------------###
        #executor_url = base64.b64decode(worker_id).decode()
        ###-------------------------------------------------------###

        ###-----------------Get Embedding Base List-----------------###
        # EmbeddingBase = DBQ.loadEmbeddingList(dbh, request.TenantID)
        # EmbeddingBase = [item[0] for item in EmbeddingBase]
        #EmbeddingBase = dbh.GetEmbeddingBase(request.TenantID, request.EmbeddingID)
        EmbeddingString = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
        RespPoll = []
        MaxWorker = 1
        MinRecordPerWorker = 1000
        if (len(executor_ids) > MaxWorker):
            MaxWorker = len(executor_ids)
            #Absolut Max Worker Can't be Changed Because of the Worker Pool is Limited
        # if (len(EmbeddingBase) > MinRecordPerWorker):
        #     if (int(len(EmbeddingBase) / MinRecordPerWorker)) > MaxWorker:
        #         #MaxWorker = int(len(EmbeddingBase) / MinRecordPerWorker)
        #         MinRecordPerWorker = math.ceil(len(EmbeddingBase) / MaxWorker)

        # temp_list = []
        # EmbeddingString = [EmbeddingBase[i:i + MinRecordPerWorker] for i in range(0, len(EmbeddingBase), MinRecordPerWorker)]
        x_val = 0
        # for i, val in enumerate(EmbeddingBase):
        #     x_val = i
        #     if i == MinRecordPerWorker:
        #         temp_list.append(val[0])
        #         EmbeddingString.append(temp_list)
        #         temp_list = []
        #     if i == len(EmbeddingBase)-1:
        #         temp_list.append(val[0])
        #         EmbeddingString.append(temp_list)
        #         temp_list = []
        #     else:
        #         temp_list.append(val[0])
        ###-----------------------Save embedding feature to redis-----------------------###
        # r_proc_feature.set(str(TrxID), struct.pack('%sf' % len(request.EmbeddingFeature), *request.EmbeddingFeature))

        ###-------------------------------------------------------###
        with concurrent.futures.ThreadPoolExecutor(max_workers=MaxWorker) as executor:
            # Start the load operations and mark each future with its URL
            future_to_exec = {executor.submit(EXE.IdentifyManyAsync, IMQ, r_req_resp, logging, executor_id, TrxID=TrxID, 
                                            TenantID=request.TenantID, 
                                            EmbeddingID=TrxID, 
                                            EmbeddingFeature=EmbeddingFeature,
                                            EmbeddingBytes=bytes('','utf-8'),
                                            Timestamp=Timestamp, DeviceID=request.DeviceID, 
                                            EmbeddingString=estring, NumResult=request.NumResult): (executor_id, estring) for executor_id, estring in zip(executor_ids, EmbeddingString)}
            
            for future in concurrent.futures.as_completed(future_to_exec):
                url = future_to_exec[future]
                try:
                    RespPoll.append(future.result())
                except Exception as ex:
                    logging.info('%r generated an exception: %s' % (url, ex))
                # else:
                #     logging.info('%r page is %d bytes' % (url, len(RespPoll)))
        ###-----------------Send Request to Worker-----------------###

        # reqs = executor_pb2.IdentifyOneRequest(TrxID=TrxID, TenantID=request.TenantID, TenantUserID=request.TenantUserID,
        #                                 EmbeddingID=request.EmbeddingID, EmbeddingFeature=request.EmbeddingFeature,
        #                                 Timestamp=Timestamp, DeviceID=request.DeviceID, EmbeddingString=EmbeddingString)
        # resp = executor_pb2_grpc.ExecutorInfranFaceIDStub(grpc.insecure_channel(executor_url)).IdentifyOne(reqs)
        ###-------------------------------------------------------###
        similarities = []
        embeding_ids = []
        for subResp in RespPoll:
            # print(subResp['Result'])
            similarities.extend([Resp['ConfidenceScore'] for Resp in subResp['Result']])
            embeding_ids.extend([Resp['MatchEmbeddingID'] for Resp in subResp['Result']])
        # print(similarities)
        # print(Counter(embeding_ids))
        # similarities = np.array(similarities)
        sim_array = np.array(similarities)
        sim_array[::-1].sort()
        id_emb_idx = []
        match_results = []
        # print(sim_array)
        for res in sim_array[:request.NumResult]:
            idx = np.where(similarities==res)[0][0]
            id_emb_idx.append(idx)
            match_results.append(infrandriver_pb2.IdentifyResult(
                MatchEmbeddingID=embeding_ids[idx], 
                ConfidenceScore=str(f"{(res*100):.3f}%")))
        # print(id_emb_idx)
        match_embeddings = [embeding_ids[idx] for idx in id_emb_idx]

        respons_obj = infrandriver_pb2.IdentifyManyResponse

        # respons_obj.MatchEmbeddingID.extend(list(match_embeddings))
        # respons_obj.ConfidenceScore.extend(list(sim_array[:request.NumResult]))
        df_match_embeddings = list(match_embeddings)
        # print(df_match_embeddings)
        df_similarities = list(sim_array[:request.NumResult])
        # print(df_similarities)
        
        max_val_resp = RespPoll[-1]
        # print(max_val_resp)
        pool_executor_response_checkpoint_time = time.time()*1000
        logging.info(f'Concurent All Executor time : {round(pool_executor_response_checkpoint_time-get_executor_checkpoint_time)} miliseconds')
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" IdentifyOne Request Finished\n")
        logging.info("------------------------------------------------------------------------------")
        logging.info("\n")

        ###-----------------------Remove embedding feature to redis-----------------------###
        # r_proc_feature.delete(TrxID)
        
        return infrandriver_pb2.IdentifyManyResponse(TrxID=TrxID, Result="Get Response",
                                                    Status=max_val_resp['Status'], ErrCode=str(max_val_resp['ErrCode']), EmbeddingID=max_val_resp['EmbeddingID'],
                                                    MatchEmbeddingID=df_match_embeddings, 
                                                    ConfidenceScore=df_similarities,
                                                    Message=max_val_resp['Message'], Timestamp=Timestamp,
                                                    MatchingResult=match_results)

    def RegisterUser(self, request, context):
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" Get RegisterUser Request")
        TrxID = request.TrxID
        ImgData = request.ImgData
        UserID = request.UserID
        UserUniqueId = "-"
        Result = "-1"
        Status = "Failed"
        ErrCode = "999"
        Message = "Internal Server Error"
        Timestamp = request.Timestamp
        BypassVerification = False
        FaceDetector = request.FaceDetector
        # FullName = request.FullName if hasattr(request, "FullName") else None
        # Nik = request.Nik if request.Nik and request.Nik.strip() != "" else None
        # BirthDate = request.BirthDate if (hasattr(request, "BirthDate") and request.BirthDate) else None
        # Gender = request.Gender if hasattr(request, "Gender") else None
        if(request.FaceDetector==9):
            FaceDetector = 0
            BypassVerification = True
        if(request.TrxID==''):
            TrxID = str(uuid.uuid4())
        if (request.Timestamp==''):
            Timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        if(request.TenantID==''):
            ErrCode = "011"
        if(request.rkey==''):
            embedding_data = IMQ.RedisHandler(db=2)
            poolget = WKP.PoolGetter(r_worker_pool)
            if(FaceDetector==0):
                ###-----------------Get Face Detector-----------------###
                start_facedetect = time.time()*1000
                resp_face_detector = EXE.FaceDetectorFunc(ImgData, TrxID, Timestamp)
                ImgData = resp_face_detector['ImgWarp']
                finish_facedetect = time.time()*1000
                logging.info(f'Face Detection time : {round(finish_facedetect-start_facedetect)} miliseconds')

            ###-----------------Get Feature Extraction-----------------###
            start_featureextract = time.time()*1000
            resp_feature_extractor = EXE.FeatureExtractorFunc(ImgData, TrxID, Timestamp)
            Embedding = resp_feature_extractor['Embedding']
            Embedding = np.load(io.BytesIO(Embedding), allow_pickle=True)
            embedding_l = pd.DataFrame(Embedding).values[0]
            finish_featureextract= time.time()*1000
            logging.info(f'Feature Extractor time : {round(finish_featureextract-start_featureextract)} miliseconds')
            ###-----------------------Identify One-----------------------###
            resp_identify = self.IdentifyOne(infrandriver_pb2.IdentifyOneRequest(TrxID=TrxID, TenantID=request.TenantID, 
                                                                                Timestamp=Timestamp, DeviceID=request.DeviceID, 
                                                                                EmbeddingFeature=embedding_l)
                                            , context)
            logging.info(f"Confidence Score : {resp_identify.ConfidenceScore}")
            if resp_identify.ConfidenceScore <= 0.8 and not BypassVerification:
                logging.info(f"Normal Registration")
                ###-----------------Register-----------------###
                start_register = time.time()*1000
                # Register to Databases
                current_date = datetime.now().strftime("%Y/%m/%d")
                # Register to Database
                memberInfo = DBQ.registerUserMember(dbh, request.TenantID, request.Name, request.Name, request.Password, request.UserID, request.Description)
                UserID, UserID_unique, Name = memberInfo[0][0], memberInfo[0][2], memberInfo[0][3]
                save_path = f"member_data/{request.TenantID}/{current_date}/{UserID}"
                UserUniqueId = str(uuid.uuid4())
                rkey = "{0}#{1}#{2}".format(request.TenantID, UserID, UserUniqueId)
                imageInfo = DBQ.saveImage(dbh, request.TenantID, UserID, save_path, rkey, base64.b64decode(ImgData))
                embeddingInfo = DBQ.saveEmbedding(dbh, request.TenantID, UserID, save_path, rkey, Embedding.dumps(), embedding_l.tolist())
                logging.info(f"Registered to Databases {rkey}")
                # Register to Redis
                embedding_data.set(rkey, Embedding.dumps())
                logging.info(f"Registered to Redis {rkey}")
                #Register to Executor
                poolget = WKP.PoolGetter(r_worker_pool)
                worker_id = poolget.GetAllWorker('executor_pool-%s-'%(local_host_name))
                executor_id = ''
                last_idx_embedding = 0
                for i in worker_id:
                    if int(str(i).split('-')[-1])>last_idx_embedding:
                        last_idx_embedding = int(str(i).split('-')[-1])
                        executor_id = str(i).replace('executor_pool-%s-'%(local_host_name),'')
                logging.info(f"Executor ID: {executor_id}")
                resp_register_exeutor = EXE.RegisterUserAsync(IMQ, r_req_resp, logging, executor_id, TrxID=TrxID, 
                                                          TenantID=request.TenantID, EmbeddingID=rkey, 
                                                          Timestamp=Timestamp, DeviceID=request.DeviceID)
            
                logging.info(f'Register Response Executor: {resp_register_exeutor}')
                Result = resp_register_exeutor['Result']
                ErrCode = str(resp_register_exeutor['ErrCode'])
                Message = resp_register_exeutor['Message']
                Status = "Success" if ErrCode=="0" else "Failed"
                
                finish_register= time.time()*1000
                logging.info("Registered to Executor")
                logging.info(f'Register to Executor time : {round(finish_register-start_register)} miliseconds')
          
            elif BypassVerification:
                logging.info(f"Bypass Verification on user {request.TenantID}|{UserID}")
                ###-----------------Register-----------------###
                start_register = time.time()*1000
                # Register to Databases
                current_date = datetime.now().strftime("%Y/%m/%d")

                # Register to Database
                memberInfo = DBQ.loadMemberUser(dbh,request.TenantID, UserID)
                if len(memberInfo) < 1:
                    Result = "Failed"
                    ErrCode = "404"
                    Message = "User Not Found"
                    Status = "Failed"
                else:
                    UserID, UserID_unique, Name = memberInfo[0][0], memberInfo[0][2], memberInfo[0][3]
                    save_path = f"member_data/{request.TenantID}/{current_date}/{UserID}"
                    # Split the string by '_'
                    UserUniqueId = UserID_unique.split("_")

                    # Check if the last element is a number
                    if len(UserUniqueId) > 1 and UserUniqueId[-1].isdigit():
                        # If the last element is a number, increment it
                        new_sequence = int(UserUniqueId[-1]) + 1
                        UserUniqueId[-1] = str(new_sequence)
                    else:
                        # If the last element is not a number, append "_0"
                        UserUniqueId.append("0")

                    # Reconstruct the string
                    UserID_unique = "_".join(UserUniqueId)
                    UserUniqueId = str(uuid.uuid4())
                    rkey = "{0}#{1}#{2}".format(request.TenantID, UserID, UserUniqueId)
                   
                    imageInfo = DBQ.saveImage(dbh, request.TenantID, UserID, save_path, rkey, base64.b64decode(ImgData))
                    embeddingInfo = DBQ.saveEmbedding(dbh, request.TenantID, UserID, save_path, rkey, Embedding.dumps(), embedding_l.tolist())
                    logging.info(f"Registered to Databases {rkey}")
                    # Register to Redis
                    embedding_data.set(rkey, Embedding.dumps())
                    logging.info(f"Registered to Redis {rkey}")
                    #Register to Executor
                    poolget = WKP.PoolGetter(r_worker_pool)
                    worker_id = poolget.GetAllWorker('executor_pool-%s-'%(local_host_name))
                    executor_id = ''
                    last_idx_embedding = 0
                    for i in worker_id:
                        if int(str(i).split('-')[-1])>last_idx_embedding:
                            last_idx_embedding = int(str(i).split('-')[-1])
                            executor_id = str(i).replace('executor_pool-%s-'%(local_host_name),'')
                    logging.info(f"Executor ID: {executor_id}")
                    resp_register_exeutor = EXE.RegisterUserAsync(IMQ, r_req_resp, logging, executor_id, TrxID=TrxID, 
                                                          TenantID=request.TenantID, EmbeddingID=rkey, 
                                                          Timestamp=Timestamp, DeviceID=request.DeviceID)
                    logging.info(f'Register Response Executor: {resp_register_exeutor}')
                    Result = resp_register_exeutor['Result']
                    ErrCode = str(resp_register_exeutor['ErrCode'])
                    Message = resp_register_exeutor['Message']
                    Status = "Success" if ErrCode=="0" else "Failed"
                
                finish_register= time.time()*1000
                logging.info("Registered to Executor")
                logging.info(f'Register to Executor time : {round(finish_register-start_register)} miliseconds')
            else:
                logging.info(f"Fallback Registration")
                start_register = time.time()*1000
                poolget = WKP.PoolGetter(r_worker_pool)
                worker_id = poolget.GetAllWorker('executor_pool-%s-'%(local_host_name))
                executor_id = ''
                rkey = request.rkey
                last_idx_embedding = 0
                ErrCode = "1"
                Result = "Face Already Registered"
                Message = "Face Duplicate"
                UserID = resp_identify.MatchEmbeddingID
                
                finish_register= time.time()*1000
                logging.info("Registered to Executor")
                logging.info(f'Register to Executor time : {round(finish_register-start_register)} miliseconds')
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" RegisterFace Request Finished\n")
        logging.info("------------------------------------------------------------------------------")
        logging.info("\n")
        
        return infrandriver_pb2.RegisterUserResponse(TrxID=TrxID, Result=Result,
                                                    Status=Status, ErrCode=ErrCode, 
                                                    UserID=UserID, UserUniqueId=UserUniqueId,
                                                    Message=Message, Timestamp=Timestamp)
             
    def DeleteUser(self, request, context):
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
        
        rkeys = [request.rkey]
        poolget = WKP.PoolGetter(r_worker_pool)
        start_register = time.time()*1000
        if(request.rkey==''):

            embedding_data = IMQ.RedisHandler(db=2)
            ###-----------------Delete-----------------###
            # Delete to Databases
            userInfos = DBQ.loadMemberUser(dbh, request.TenantID, request.UserID)
            deleteInfo = DBQ.deleteEmbedding(dbh, request.TenantID, request.UserID, request.UserID)
            logging.info("Deleted from Databases")
            # Delete to Redis
            rkeys = DBQ.getMessageList(embedding_data, "keys {0}#{1}".format(request.TenantID, request.UserID))
            rkeys = [rkey.decode() for rkey in rkeys]
            for rkey in rkeys:
                DBQ.deleteMessage(embedding_data, rkey)
            logging.info("Deleted from redis")
            logging.info(rkeys)
            #Delete to Executor
            poolget = WKP.PoolGetter(r_worker_pool)
        ###-----------------Get Executor Destination-----------------###

        if(len(rkeys)>0):
            start_identification = time.time()*1000
            worker_id = poolget.GetAllWorker('executor_pool-%s-'%(local_host_name))
            executor_ids = []
            for i in worker_id:
                executor_ids.append(str(i).replace('executor_pool-%s-'%(local_host_name),''))
            logging.info(f"Executor ID: {executor_ids}")
            get_executor_checkpoint_time = time.time()*1000
            logging.info(f'Get All Executor time : {round(get_executor_checkpoint_time-start_identification)} miliseconds')
            MaxWorker = len(executor_ids)
            RespPoll = []
            logging.info(f"Executor ID: {executor_ids}")

            with concurrent.futures.ThreadPoolExecutor(max_workers=MaxWorker) as executor:
                # Start the load operations and mark each future with its URL
                future_to_exec = {executor.submit(EXE.DeleteUserAsync, IMQ, r_req_resp, logging, executor_id, TrxID=TrxID, 
                                                TenantID=request.TenantID, 
                                                EmbeddingIDs=rkeys,
                                                Timestamp=Timestamp, DeviceID=request.DeviceID): (executor_id) for executor_id in executor_ids}
                
                for future in concurrent.futures.as_completed(future_to_exec):
                    url = future_to_exec[future]
                    try:
                        RespPoll.append(future.result())
                    except Exception as ex:
                        logging.info('%r generated an exception: %s' % (url, ex))
                    # else:
                    #     logging.info('%r page is %d bytes' % (url, len(RespPoll)))
            ###-----------------Send Request to Worker-----------------###

            # reqs = executor_pb2.IdentifyOneRequest(TrxID=TrxID, TenantID=request.TenantID, TenantUserID=request.TenantUserID,
            #                                 EmbeddingID=request.EmbeddingID, EmbeddingFeature=request.EmbeddingFeature,
            #                                 Timestamp=Timestamp, DeviceID=request.DeviceID, EmbeddingString=EmbeddingString)
            # resp = executor_pb2_grpc.ExecutorInfranFaceIDStub(grpc.insecure_channel(executor_url)).IdentifyOne(reqs)
            ###-------------------------------------------------------###
            
            first_resp = RespPoll[0]
            Result = first_resp['Result']
            ErrCode = str(first_resp['ErrCode'])
            Message = first_resp['Message']
            Status = "Success" if ErrCode=="0" else "Failed"
        else:
            ErrCode = "1"
            Message = "UserId Not Found"
            Status = "Failed"
            rkeys = ['-']
        
        finish_register= time.time()*1000
        logging.info("Deleteed to Executor")
        logging.info(f'Delete to Executor time : {round(finish_register-start_register)} miliseconds')    

        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S.%f")+" DeleteFace Request Finished\n")
        logging.info("------------------------------------------------------------------------------")
        logging.info("\n")

        return infrandriver_pb2.DeleteUserResponse(TrxID=TrxID, Result=Result,
                                                    Status=Status, ErrCode=ErrCode,
                                                    Message=Message, Timestamp=Timestamp,
                                                    rkey=rkeys[0])

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), maximum_concurrent_rpcs=None)
    infrandriver_pb2_grpc.add_DriverInfranFaceIDServicer_to_server(DriverInfranFaceID(), server)
    server.add_insecure_port('[::]:%s'%(os.getenv('PORT')))
    server.start()
    try:  
        server.wait_for_termination()
    except Exception as e: 
        print(e)

if __name__ == '__main__':
    ##---------------Generate Worker Destination----------------##
        
    base64_string = base64.b64encode("GeeksForGeeks is the best".encode()).decode()
    reconverted = base64.b64decode(base64_string).decode()
    
    driver_id = "%s:%s"%(os.getenv('HOST'), os.getenv('PORT'))
    driver_notifier = WKP.PoolNotifier(r_worker_pool, driver_id, True, 'driver_pool-')
    driver_notifier.start()

    db_hb = DBP.DatabaseHeartBeat(dbh, logging)
    db_hb.start()

    serve()
    
