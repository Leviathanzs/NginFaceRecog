import infranlib.databases.redis_handler as IMQ
import threading
import time
import random
from dotenv import load_dotenv
import os

load_dotenv(os.path.dirname(os.path.abspath(__file__))+'/.env')

local_host_name = os.getenv('LOCAL_HOST')

r_executor_pool = IMQ.RedisHandler(db=3)


class PoolNotifier(threading.Thread):
    def __init__(self, r, executor_id,loop=True, prefix='executor_pool-%s-'%(local_host_name)):
        threading.Thread.__init__(self)
        self.redis = r
        self.loop = loop
        self.executor_id = executor_id
        self.prefix = prefix

    def run(self):
        while self.loop:
            self.redis.set(self.prefix+self.executor_id, 1, timeout=300)
            time.sleep(0.2)

class PoolGetter():
    def __init__(self, r):
        self.redis = r
    
    def GetWorker(self, prefix='executor_pool-%s-'%(local_host_name)):
        # print(self.redis.query('KEYS executor_pool-*'))
        return random.choice(self.redis.query(f'KEYS {prefix}*')).decode().replace(prefix,'')
    def GetAllWorker(self, prefix='executor_pool-%s-'%(local_host_name)):
        # print(self.redis.query('KEYS worker_pool-*'))
        return [executor_id.decode() for executor_id in self.redis.query(f'KEYS {prefix}*')]
        