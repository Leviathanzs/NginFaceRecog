import redis
import threading
import msgpack
import time
import os
from dotenv import load_dotenv
import os

load_dotenv(os.path.dirname(os.path.abspath(__file__))+'/.env')

class RedisHandler():
    
    def __init__(self, host=os.getenv('RD_HOST'), port=os.getenv('RD_PORT'), db=0, password=os.getenv('RD_PASSWORD')):
        self.pool = redis.ConnectionPool(host=host, port=port, db=db, password=password)
        self.r_conn = redis.Redis(connection_pool=self.pool)

    def get_connection(self):
        r = redis.Redis(connection_pool=self.pool)
        return r

    def check_connection(self):
        return self.r_conn.execute_command('ping')

    def query(self, command):
        if not self.check_connection():
            self.r_conn = self.get_connection()
        return self.r_conn.execute_command(command)

    def get(self, keys):
        if not self.check_connection():
            self.r_conn = self.get_connection()
        return self.r_conn.get(keys)

    def set(self, keys, value, timeout=None):
        if not self.check_connection():
            self.r_conn = self.get_connection()
        return self.r_conn.set(keys, value, px=timeout)
    
    def sadd(self, key, value):
        if not self.check_connection():
            self.r_conn = self.get_connection()
        return self.r_conn.sadd(key, value)
    
    def expire(self, keys, delta):
        if not self.check_connection():
            self.r_conn = self.get_connection()
        return self.r_conn.expire(keys, delta)   

    def delete(self, keys):
        if not self.check_connection():
            self.r_conn = self.get_connection()
        return self.r_conn.delete(keys)

    def getAlldata(self):
        if not self.check_connection():
            self.r_conn = self.get_connection()

        keys = [key.decode() for key in self.r_conn.execute_command('KEYS *')]
        return keys

    def reset(self):
        if not self.check_connection():
            self.r_conn = self.get_connection()

        keys = [key.decode() for key in self.r_conn.execute_command('KEYS *')]
        result = []
        succes = 0
        failed = 0
        for key in keys:
            try:
                self.r_conn.delete(key)
                succes += 1
            except:
                failed += 1
                pass
        return {'succes': succes, 'failed': failed}


class MessageListener(threading.Thread):
    def __init__(self, r, exec_name, channels, callback, logger, send_message=None, loop=True):
        threading.Thread.__init__(self)
        self.redis = r
        self.exec_name = exec_name
        self.pubsub = self.redis.pubsub()
        self.pubsub.subscribe(channels)
        self.channels = channels
        self.loop = loop
        self.callback = callback
        self.logger = logger
        self.send_message = send_message
        self.succes = False
        self.message = None
    
    def work(self, item):
        try:
            message = msgpack.unpackb(item['data'], raw=False)
        except:
            self.logger.error("Error: ", item)
            return
        self.logger.info('Channel: {} - Message Length: {}'.format(item['channel'], len(message)))
        self.message = message
        if self.callback:
            self.callback(self.channels, message, self.redis, self.exec_name)
        
    def GetMessage(self):
        return self.message

    def run(self):
        for item in self.pubsub.listen():
            if item['type'] == 'message':
                self.work(item)
                if not self.loop:
                    break
            else:
                self.logger.info('Redis : {}'.format(item))
                if item['type'] == 'unsubscribe':
                    break
                elif item['type'] == 'subscribe' and item['data'] == 1:
                    if self.send_message:
                        self.send_message.send()
                elif item['type'] == 'subscribe' and item['data'] == 0:
                    break
                elif item['data'] == 1:
                    continue
    
    def ForceStop(self):
        try:
            self.pubsub.unsubscribe(self.channels)
        except Exception as e1:
            self.logger.error('Error: {}'.format(e1))
            pass
        try:
            self._tstate_lock.release()
            self._stop()
        except Exception as e2:
            self.logger.error('Error: {}'.format(e2))
            pass    
    
    def WaitForResult(self, timeout=10000, interval=0.003):
        "Method to wait for the result of the thread, Use in Middleware"
        cnt = timeout/(1000*interval)
        while self.message is None and cnt > 0:
            time.sleep(interval)
            cnt -= 1
        self.ForceStop()
        if self.message is None:
            return {
                'TrxID': '0000',
                'Message': 'Error Executor', 
                'Result':'Timeout response from executor', 
                'ErrCode': 1,
                'Status': 'Timeout response from executor',
                'EmbeddingID': '',
                'MatchEmbeddingID': 'id_embedding',
                'ConfidenceScore': 0.0,
                'Timestamp': ''
            }
        return self.message
        

class MessagePublisher():
    def __init__(self, r, channels, message, logger):
        self.redis = r
        self.channels = channels
        self.message = message
        self.logger = logger

    def pub(self, name, value):
        self.redis.publish(name, value)
    
    def send(self):
        self.redis.publish(self.channels, msgpack.packb(self.message, use_bin_type=True))

class MessageBuilder():
    def __init__(self):
        pass

    def build_verify_message_request(self, tenant, facedetector, img_data):
        return {
            "tenant": tenant,
            "face_detector": facedetector,
            "img_data": img_data
        }