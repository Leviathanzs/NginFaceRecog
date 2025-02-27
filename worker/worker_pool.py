import infranlib.databases.redis_handler as IMQ
import threading
import time
import random
import uuid

r_worker_pool = IMQ.RedisHandler(host='127.17.0.1', port=6379, db=3)
class PoolNotifier(threading.Thread):
    def __init__(self, r: IMQ.RedisHandler, worker_id=str(uuid.uuid4()),loop=True, prefix='executor_pool-'):
        threading.Thread.__init__(self)
        self.redis = r
        self.loop = loop
        self.worker_id = worker_id
        self.prefix = prefix

    def run(self):
        while self.loop:
            self.redis.set(self.prefix+self.worker_id, 1, timeout=300)
            time.sleep(0.2)

class PoolGetter():
    def __init__(self, r: IMQ.RedisHandler):
        self.redis = r
    
    def GetWorker(self, prefix='executor_pool-'):
        # print(self.redis.query('KEYS worker_pool-*'))
        return random.choice(self.redis.query(f'{prefix}*')).decode()
    def GetAllWorker(self, prefix='executor_pool-'):
        # print(self.redis.query('KEYS worker_pool-*'))
        return [url_executor_b64.decode() for url_executor_b64 in self.redis.query(f'{prefix}*')]
    
        