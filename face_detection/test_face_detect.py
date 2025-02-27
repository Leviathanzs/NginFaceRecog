import logging

from face_detector_core import WKP, IMQ
from read_config import readConfig

data_facedetect = readConfig()

r_facedetect_pool = IMQ.RedisHandler(db=3)
facedetect_id = data_facedetect['face_detector_id']
TrxID = '1111'

with open('test_ImgData.txt', 'r') as ImgData:

    req = {
        'TrxID': TrxID,
        'Method': 'MTCNN',
        'ImgData': ImgData.read()
    }

messagep = IMQ.MessagePublisher(r_facedetect_pool.get_connection(), facedetect_id, req, logging)
# print('ke message listener')
client = IMQ.MessageListener(r_facedetect_pool.get_connection(), facedetect_id, TrxID, None, logging, send_message=messagep)
client.start()
# print('ke response')
response = client.WaitForResult()

print(response)
