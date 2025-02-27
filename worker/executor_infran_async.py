import datetime
import logging
import time
import msgpack

import core_executor

import csv
import os
from inspect import getsourcefile

HomeDir = os.path.dirname(os.path.abspath(getsourcefile(lambda:0)))
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(filename=f'{HomeDir}/infran-executor-{datetime.datetime.now().strftime("%Y%m%dT%H%M%S")}.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    force=True,
                    level=logging.INFO)
exec_chan = '000000000000000'

class ExecutorInfran:
    def __init__(self, channel, message, redis_conn, exec_name):
        logging.info("ExecutorInfran is initialized from channel %s" % channel)
        self.exec_name = exec_name
        self.redis_conn = redis_conn
        if (message['Method'] == 'IdentifyFace'):
            self.IdentifyFace(message)
        elif (message['Method'] == 'IdentifyManyFace'):
            self.IdentifyManyFace(message)
        elif (message['Method'] == 'RegisterFace'):
            self.RegisterFace(message)
        elif (message['Method'] == 'DeleteFace'):
            self.DeleteFace(message)
        # elif (message['method'] == 'VerifyFace'):
        #     self.VerifyFace(message)
        # elif (message['method'] == 'RegisterUser'):
        #     self.RegisterUser(message)
        # elif (message['method'] == 'RegisterUserMember'):
        #     self.RegisterUserMember(message)
        else:
            self.NotFound(message)
            
    def IdentifyFace(self, request, context=None):
        global exec_chan
        logging.info("==== Start IdentifyFace ====")
        
        filename = "./Result_Worker(tessar).csv"
        fields = ['Waktu Face Matching', 'Confidence Score', 'Matching Embedding ID']
        id_embedding = ['00000000-0000-0000-0000-000000000000#00000000-0000-0000-0000-000000000000']
        
        logging.info(f"Received request: {request}")

        if request['TenantID'] == '':
            message = "Error Request: tenant is Undefined"
            err_code = 1
            logging.error(message)
        elif request['EmbeddingID'] == '' and request['EmbeddingBytes'] == bytes('', 'utf-8'):
            message = "Error Request: Feature is Undefined"
            err_code = 1
            logging.error(message)
        else:
            if 'EmbeddingFeature' in request.keys():
                EmbeddingFeature = request['EmbeddingFeature']
            else:
                EmbeddingFeature = []
            
            logging.info("Starting identification process...")

            startt = time.time() * 1000
            try:
                logging.info("Calling core_executor.identification_dot_product()...")
                similarity, id_embedding, err_code = core_executor.identification_dot_product(
                    self.exec_name,
                    request['EmbeddingID'],
                    EmbeddingFeature,
                    request['EmbeddingBytes'],
                    list(request['EmbeddingString'])
                )
                logging.info(f"Identification result: Similarity={similarity}, ID_Embedding={id_embedding}, ErrCode={err_code}")

                if len(similarity) == 0:
                    similarity = [0]
                    id_embedding = ['0000000000000']
                    logging.warning("No match found. Assigning default values.")

                message = 'Success get result'
            
            except Exception as err:
                logging.error(f"Error in identification_dot_product: {err}")
                similarity = [0]
                id_embedding = ['']
                message = "Error Identification Dot Product"
                err_code = 1

            endt = time.time() * 1000
            duration = round(endt - startt)
            logging.info(f"Identification dot product time server: {duration} milliseconds")

            # Log the execution result before saving to CSV
            logging.info(f"Identification completed. Duration: {duration} ms, Similarity: {similarity[0]}, ID: {id_embedding[0]}")

            # Try to save the result to a CSV file (optional)
            try:
                rows = [duration, similarity[0], id_embedding[0]]
                if os.path.exists(filename): 
                    with open(filename, 'a') as csvfile: 
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(rows)
                        logging.info(f"Successfully wrote to CSV: {filename}")
                else: 
                    with open(filename, 'w') as csvfile: 
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow(fields)
                        csvwriter.writerow(rows)
                        logging.info(f"CSV created and wrote first entry: {filename}")
            except Exception as csv_error:
                logging.error(f"Error writing to CSV: {csv_error}")

        # Prepare response data
        data = {
            'TrxID': request['TrxID'],
            'ExecutorID': exec_chan,
            'Message': message,
            'Result': 'Get response from executor',
            'ErrCode': err_code,
            'Status': 'Get response from executor',
            'EmbeddingID': request['EmbeddingID'],
            'MatchEmbeddingID': id_embedding[0] if id_embedding else '',
            'ConfidenceScore': similarity[0] if similarity else 0,
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }

        logging.info(f"Identification result executor {exec_chan}: {id_embedding} {similarity}")
        
        # Publish result to Redis
        try:
            logging.info(f"Publishing result to Redis on channel {request['ReturnChannel']}...")
            self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))
            logging.info("Successfully published result to Redis.")
        except Exception as redis_error:
            logging.error(f"Failed to publish to Redis: {redis_error}")

        logging.info("==== End IdentifyFace ====")

    def IdentifyManyFace(self, request, context=None):
        global exec_chan
        logging.info("IdentifyManyFace is called.")

        if(request['TenantID']==''):
            
            message = "Error Request: tenant is Undefined"
            err_code = 1
            result = []
        elif(request['EmbeddingID']=='' and request['EmbeddingBytes']==bytes('','utf-8')):
            
            message = "Error Request: Feature is Undefine is Undefined"
            err_code = 1
            result = []
        else:
            result = []
            if 'EmbeddingFeature' in request.keys():
                # EmbeddingFeature = core_executor.struct.unpack('512f', request['EmbeddingFeature'])
                EmbeddingFeature = request['EmbeddingFeature']
            else:
                EmbeddingFeature = []
            startt = time.time() * 1000
            try:
                similarity, id_embedding, err_code = core_executor.identification_dot_product(self.exec_name, request['EmbeddingID'], EmbeddingFeature, request['EmbeddingBytes'], list(request['EmbeddingString']), request['NumResult'])
                message = 'Succes get result'
                for i, sim in enumerate(similarity):
                    result.append({
                        'MatchEmbeddingID': id_embedding[i],
                        'ConfidenceScore': sim,
                    })
            except Exception as err:
                logging.error(err)
                
                similarity = 0
                id_embedding = ''
                message = "Error Identification Dot Product"
                err_code = 1

            endt = time.time() * 1000
            duration = round(endt - startt)
            logging.info(f"Identification dot product time server: {duration} milliseconds")

        data = {
            'TrxID': request['TrxID'],
            'ExecutorID': exec_chan,
            'Message': message, 
            'Result': result, 
            'ErrCode':err_code,
            'EmbeddingID': request['EmbeddingID'],
            'Status': 'Get response from executor',
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }
        logging.info(f"Identification dot product result executor {exec_chan} : {id_embedding} {similarity}")
        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

    def RegisterFace(self, request, context=None):

        logging.info("RegisterFace is called.")

        if(request['TenantID']==''):
            
            message = "Error Request: tenant is Undefined"
            err_code = 1
        elif(request['EmbeddingID']==''):

            message = "Error Request: embeding id is Undefined"
            err_code = 1
        else:
            startt = time.time() * 1000
            try:
                message = core_executor.register_feature(self.exec_name, request['EmbeddingID'])
                err_code = 0
            except Exception as err:
                logging.error(err)
                
                message = "Error Registration"
                err_code = 1

            endt = time.time() * 1000
            duration = round(endt - startt)
            logging.info(f"Registration time server: {duration} milliseconds")

        data = {
            'TrxID': request['TrxID'],
            'ExecutorID': exec_chan,
            'Message': message, 
            'Result':'Get response from executor', 
            'ErrCode':err_code,
            'Status': 'Get response from executor',
            'EmbeddingID': request['EmbeddingID'],
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }
        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

    def DeleteFace(self, request, context=None):

        logging.info("DeleteFace is called.")

        if(request['TenantID']==''):
            
            message = "Error Request: tenant is Undefined"
            err_code = 1
        elif(request['EmbeddingIDs']==''):

            message = "Error Request: embeding id is Undefined"
            err_code = 1
        else:
            startt = time.time() * 1000
            try:
                message = core_executor.remove_feature(self.exec_name, request['EmbeddingIDs'])
                err_code = 0
            except Exception as err:
                logging.error(err)
                
                message = "Error Delete"
                err_code = 1

            endt = time.time() * 1000
            duration = round(endt - startt)
            logging.info(f"Delete time server: {duration} milliseconds")

        data = {
            'TrxID': request['TrxID'],
            'ExecutorID': exec_chan,
            'Message': message, 
            'Result':'Get response from executor', 
            'ErrCode':err_code,
            'Status': 'Get response from executor',
            'EmbeddingID': request['EmbeddingIDs'],
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }
        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

    def NotFound(self, request, context=None):
        logging.info(f"Method not found {request['Method']}")
        data = {'TrxID': request['TrxID'],
            'Message': 'Method not found', 
            'Result':'Get response from executor', 
            'ErrCode':1,
            'Status': 'Get response from executor',
            'EmbeddingID': request['EmbeddingID'],
            'MatchEmbeddingID': 'id_embedding',
            'ConfidenceScore': 0,
            'Timestamp': datetime.datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        }
        self.redis_conn.publish(request['ReturnChannel'], msgpack.packb(data, use_bin_type=True))

def serve(driver_host, worker_id, executor_id, exec_name, first_embedding_number, last_embedding_number):
    global exec_chan
    r_req_resp = core_executor.IMQ.RedisHandler(db=0)
    r_executor_pool = core_executor.IMQ.RedisHandler(db=3)

    start = time.time()
    core_executor.loadAllLocalModel(executor_id, first_embedding_number, last_embedding_number)
    end = time.time()
    logging.info(f"{executor_id} - [*] load from memory init_embeddings(): Done in {end-start} seconds")

    channel_id = f"{worker_id}-{executor_id}-{str(first_embedding_number)}-{str(last_embedding_number)}"
    exec_chan = channel_id

    client = core_executor.IMQ.MessageListener(r_req_resp.get_connection(), executor_id, channel_id, ExecutorInfran, logging)
    client.start()
    logging.info(f'{executor_id} - Message Listener is on')
    # r_command.set('worker_pool-xxxx', 1)
    client2 = core_executor.WKP.PoolNotifier(r_executor_pool, channel_id, prefix="executor_pool-%s-"%(driver_host))
    logging.info(f"{executor_id} - Mendaftarkan executor ke Redis di pool executor_pool-{driver_host}")
    client2.start()
    logging.info(f'{executor_id} - Executor Pool Notifier is on')

if __name__ == "__main__":
    
    serve('123456789', 'worker_001_exec_001', 0, 10000)