import torch
import torchvision.transforms as transforms
import cv2  
import io
import logging
import numpy as np
from PIL import Image

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embeddings import get_embedding

from backbone import Backbone
from align.align_trans import (
    get_reference_facial_points,
    warp_and_crop_face,
)
from align.detector import detect_faces
from align.get_nets import PNet, RNet, ONet

import databases.database_query as DBQ
import databases.redis_handler as IMQ
import databases.postgres_handler as DBP

emb_dict = None
device = None
backbone = None
transform = None
conn = None
detector = None
reference = None
input_size = None
crop_size = None
pnet = None
rnet = None
onet = None
embedding_size = 512
input_size=[112, 112]
num_embeddings_executor = 1000


def init_model(gpu):

    global device
    global backbone
    global transform
    global input_size

    if(gpu==1):
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            message_gpu = 'Using GPU'
        else:
            message_gpu = 'Using CPU'
    else:
        device = torch.device("cpu")
        message_gpu = 'Using CPU'

    backbone = Backbone(input_size)
    backbone.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__))+"/checkpoint/backbone_ir50_ms1m_epoch120.pth", map_location=device))
    backbone.to(device)
    backbone.eval()

    transform = transforms.Compose(
        [
            transforms.Resize([int(128 * input_size[0] / input_size[1]), int(128 * input_size[0] / input_size[1])]),
            transforms.CenterCrop([input_size[0], input_size[1]]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ],
    )
    logging.info(f'[*] init_model(): {message_gpu} Done')

def get_reference():

    global reference
    global input_size
    global crop_size

    input_size = [112, 112]
    crop_size = 112
    scale = crop_size / 112.0
    reference = get_reference_facial_points(default_square=True) * scale
    
def mtcnn_detector():

    global pnet
    global rnet
    global onet

    pnet = PNet()
    rnet = RNet()
    onet = ONet()
    onet.eval()


def init_embeddings(dbh, redis):
    global transform
    global device
    global backbone
    global num_embeddings_executor

    user_list = DBQ.loadUserList(dbh)
    for user in user_list:
        user_uuid = user[0]
        # print("user_uuid: {0}".format(user_uuid))
        logging.info(f"Prepare Load Model for User: {user_uuid} to Memory")
        member_list = DBQ.loadMemberUserList(dbh, user_uuid)
        for i, member in enumerate(member_list):
            member_uuid = member[0]
            member_unique_id = member[1]
            # print("member_uuid: {0}".format(member_uuid))
            embedding_list = DBQ.loadEmbeddingList(dbh, user_uuid, member_uuid)
            image_list = DBQ.loadImageList(dbh, user_uuid, member_uuid)
            # embeddings = np.zeros([(len(embedding_list)), embedding_size])
            embedding_data = None
            if len(embedding_list) > 0:
                print("Embedding Exists")
                for idx, embedding in enumerate(embedding_list):
                    # print("embedding: {0}".format(embedding))
                    embedding_data = DBQ.loadEmbedding(dbh, embedding[0])
                    # embeddings[idx, :] = embedding_data[0][-1]
                    rkey = "{0}#{1}#{2}#{3}".format(embedding_data[0][0], embedding_data[0][1], embedding_data[0][3], member_unique_id)
                    # print("rkey: {0}".format(rkey))
                    rst = redis.set(rkey, embedding_data[0][-1])
                    logging.info(f"Load Model with Memory Tag: {rkey} {rst}")
                # embeddings = np.load(io.BytesIO(byte_numpy), allow_pickle=True)

            elif len(image_list) > 0 and len(embedding_list) < 1:
                # image_list = loadImageList(dbh, user_uuid)
                # embeddings = np.zeros([(len(image_list)), embedding_size])
                print("Embedding Not Exists")
                embeddings = np.zeros([(len(image_list)), embedding_size])
                for idx, image in enumerate(image_list):
                    image_data = DBQ.loadImage(dbh, image[0])
                    # print("image_data: {0}".format(bytes(image_data[0][-1])))
                    # with open ('{0}.jpg'.format(image_data[0][0]), 'wb') as f:
                    #     f.write(bytes(image_data[0][-1]))
                    img_warp, err_code = face_alignment(Image.open(io.BytesIO(bytes(image_data[0][-1]))))
                    # print(img_warp)
                    if err_code == "000":
                        embeddings[idx, :] = get_embedding(backbone, img_warp, input_size, transform, device)
                rkey = "{3}#{0}#{1}#{2}".format(image_data[0][0], image_data[0][1], member_unique_id)
                DBQ.save_embedding(dbh, user_uuid, member_uuid, rkey, embeddings.dumps())
                # print("rkey: {0}".format(rkey))
                embedding_list = DBQ.loadEmbeddingList(dbh, user_uuid, member_uuid)
                for idx, embedding in enumerate(embedding_list):
                    embedding_data = DBQ.loadEmbedding(dbh, embedding[0])
                    # embeddings[idx, :] = embedding_data[0][-1]
                
                rkey = "{0}#{1}#{2}".format(embedding_data[0][0], embedding_data[0][1], member_unique_id, str(i//num_embeddings_executor))
                # print("rkey: {0}".format(rkey))
                rst = redis.set(rkey, embedding_data[0][-1])
                logging.info(f"Load Model with Memory Tag: {rkey} {rst}")
                # embeddings = np.load(io.BytesIO(byte_numpy), allow_pickle=True)
                        
            else:
                print("No image for user: {0} member:{1}".format(user_uuid, member_unique_id))

def loadAllLocalModel(dbh, redis, executor_number):
    print("Loading Model to cache")
    global emb_dict
    user_list = DBQ.loadUserList(dbh)
    # if emb_dict == None:
    emb_dict = dict()
    for user in user_list:
        user_uuid = user[0]
        # if user_uuid not in emb_dict:
        emb_dict[user_uuid] = dict()
        # print("user_uuid: {0}".format(user_uuid))
        member_list = DBQ.loadMemberUserList(dbh, user_uuid)
        all_embeddings_list = []
        all_member_list = []
        for member in member_list:
            member_uuid = member[0]
            # if member_uuid not in emb_dict[user_uuid]:
            #     emb_dict[user_uuid][member_uuid] = None
            rkey = "KEYS {2}#{0}#{1}*".format(user_uuid, member_uuid, executor_number)
            # print(rkey)
            embedding_key_list = redis.query(rkey)
            # print(embedding_key_list)
            for embedding_key in embedding_key_list:
                embedding_key = embedding_key.decode("utf-8")
                embeddings = redis.get(embedding_key)
                embeddings = np.load(io.BytesIO(embeddings), allow_pickle=True)
                # print(embeddings.shape)
                for embedding in embeddings:
                    all_embeddings_list.append(embedding)
                    all_member_list.append(embedding_key.split("#")[1])

        # print(len(all_embeddings_list))
        # print(all_member_list)
                
        # all_embeddings_list.append(embeddings[idx2, :])
        # emb_dict[user_uuid][member_uuid] = np.load(io.BytesIO(embeddings), allow_pickle=True)
        # print(embeddings)
        # all_embeddings = np.zeros([(len(all_embeddings_list)), embedding_size])
        # for idx, embedding in enumerate(all_embeddings_list):
        #     all_embeddings[idx, :] = embedding
        
        emb_dict[user_uuid]['all_embeddings'] = np.array(all_embeddings_list)
        emb_dict[user_uuid]['all_member_list'] = all_member_list

        # print(identification(dbh, user_uuid, Image.open("./zzz.jpg")))

def face_alignment(image):

    global reference
    global crop_size

    img_warp = None
    landmarks = []
    message = "Success"
    try:
        tm = cv2.TickMeter()
        tm.start()
    
        with torch.no_grad():
            _, landmarks = detect_faces(image, pnet, onet, rnet)

        tm.stop()
        logging.info(f"MTCNN face detection time: {tm.getTimeMilli()}")
        tm.reset()
    except Exception as err:
        message = "Error Face Detector"
        logging.info(f"face_alignment(): {err}")
    if (len(landmarks) == 0):
        message = "No Face Detected"
    else:
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        tm.start()
        warped_face = warp_and_crop_face(
            np.array(image),
            facial5points,
            reference,
            crop_size=(crop_size, crop_size),
        )
        img_warp = Image.fromarray(warped_face)

    return img_warp, message

def identification(user_uuid, image, face_detector=True, threshold=0.75):
    global emb_dict
    p_code = 1
    embeddingst = emb_dict.get(user_uuid).get('all_embeddings')
    message = "Success"
    sim = 0.0
    if face_detector:
        img_warp, message = face_alignment(image)
    if message == "Success":
        embedding_c = get_embedding(backbone, img_warp, input_size, transform, device)
        embedding_c = embedding_c.detach().numpy()
        similarity = np.dot(embedding_c, embeddingst.T)
        similarity = similarity.clip(min=0, max=1)

        sim = np.max(similarity[0])
        idx_sim = np.where(similarity[0]==sim)[0][0]

    if sim < threshold:
        result = {
            'sim': sim,
            'member_uuid': ''
        }
    else:
        result = {
            'sim': sim,
            'member_uuid': emb_dict[user_uuid]['all_member_list'][idx_sim]
        }
        p_code = 0

    return result, message, p_code


if __name__ == "__main__":
    # mtcnn_detector()
    # get_reference()
    # init_model(1)
    dbh = DBP.DatabaseHandler()
    redis = IMQ.RedisHandler(db=2)
    print(redis.check_connection())
    init_embeddings(dbh, redis)
    # loadAllLocalModel(dbh, redis,1)

    data_store = IMQ.RedisHandler(db=1)
    output = io.BytesIO()
    im = Image.open(os.path.dirname(os.path.abspath(__file__))+'/test.jpg')
    im.save(output, format=im.format)

    data_store.set('test-keys', output.getvalue())


