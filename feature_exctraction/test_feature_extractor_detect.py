import logging


from feature_extractor_core import WKP, IMQ
from read_config import readConfig

data_feature_extract = readConfig()


r_facedetect_pool = IMQ.RedisHandler(db=3)
featureextract_id = data_feature_extract['feature_extractor_id']
TrxID = '1111'

with open('test_ImgData.txt', 'r') as ImgData:

    req = {
        'TrxID': TrxID,
        'Method': 'feature',
        'ImgData': ImgData.read(),
        'FaceDetector': 0
    }

messagep = IMQ.MessagePublisher(r_facedetect_pool.get_connection(), featureextract_id, req, logging)
# print('ke message listener')
client = IMQ.MessageListener(r_facedetect_pool.get_connection(), featureextract_id, TrxID, None, logging, send_message=messagep)
client.start()
# print('ke response')
response = client.WaitForResult()

print(response)
