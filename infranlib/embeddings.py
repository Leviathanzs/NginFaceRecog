# Original code
# https://github.com/ZhaoJ9014/face.evoLVe.PyTorch/blob/master/util/extract_feature_v1.py

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn.functional as F
import torchvision.transforms as transforms
from backbone import Backbone
from tqdm import tqdm
from PIL import Image


def get_embeddings(data_root, backbone, transform, device, input_size=[112, 112], embedding_size=512):

    # check data and model paths
    assert os.path.exists(data_root)
    print(f"Data root: {data_root}")

    # define data loader
    dataset = datasets.ImageFolder(data_root, transform)
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
    )
    print(f"Number of classes: {len(loader.dataset.classes)}")
    
    # get embedding for each face
    length_dataset = len(loader.dataset)
    print("get_embeddings(): length_dataset: " + str(length_dataset))

    embeddings = np.zeros([len(loader.dataset), embedding_size])
    with torch.no_grad():
        for idx, (image, _) in enumerate(
            tqdm(loader, desc="Create embeddings matrix", total=len(loader)),
        ):
            embeddings[idx, :] = F.normalize(backbone(image.to(device))).cpu()

    # get all original images
    images = []
    for img_path, _ in dataset.samples:
        print("get_embeddings(): img_path: " + img_path)
        img = cv2.imread(img_path)
        images.append(img)

    list_folder = next(os.walk(data_root))[1]
    list_nama = []
    try:
        for folder in list_folder:
            for i in range(len(os.listdir(data_root+folder))):
                list_nama.append(folder)
    except Exception as err:
        print(err)

    print("get_embeddings(): list_nama " + str(list_nama))
    return images, embeddings, list_nama

def get_embedding(backbone, image, input_size, transform, device):

    image_tensor = transform(image)

    with torch.no_grad():
        embedding = F.normalize(backbone(image_tensor.unsqueeze(0).to(device))).cpu()

    return embedding

def get_embeddings_person(data_root, backbone, transform, device, input_size=[112, 112], embedding_size=512):
    
    assert os.path.exists(data_root)

    embeddings = np.zeros([(len(os.listdir(data_root))), embedding_size])

    for idx, image_name in enumerate(os.listdir(data_root)):

        if image_name.split('.')[-1] == 'jpg':
            str1 = '{}/{}'.format(data_root, image_name)
            # print("get_embeddings_person(): " + str1)
            image = Image.open('{}/{}'.format(data_root, image_name))

            embeddings[idx, :] = get_embedding(backbone, image, input_size, transform, device)

    return embeddings

def get_embeddings_m(data_root, backbone, transform, device, input_size=[112, 112], embedding_size=512):

    # check data and model paths
    assert os.path.exists(data_root)
    print(f"Data root: {data_root}")

    # define data loader
    dataset = datasets.ImageFolder(data_root, transform)
    loader = data.DataLoader(
        dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0,
    )
    print(f"Number of classes: {len(loader.dataset.classes)}")
    
    # get embedding for each face
    length_dataset = len(loader.dataset)
    print("get_embeddings_m(): length_dataset: " + str(length_dataset))

    embeddings = np.zeros([len(loader.dataset), embedding_size])
    print("")
    print("")

    # todo: 2021-02-22
    # compute tensor only on the new image file
    # append the new tensor to embeddings    
    
    with torch.no_grad():
        # i = 0
        for idx, (image, _) in enumerate(loader):
            # tqdm(loader, desc="Create embeddings matrix", total=len(loader)),
                # tqdm(loader, desc="Create embeddings matrix", total=len(loader)),
            # ):
            embeddings[idx, :] = F.normalize(backbone(image.to(device))).cpu()
            # print("i: " + str(i) + ", image file: " + str(dataset[i]))
            # print("i: " + str(i) + ", image file: " + str(test1))
            # print("i: " + str(i) + ", image file: " + str(image))
            # i += 1

            image_fname, _ = loader.dataset.samples[idx]
            print("image_fname: " + image_fname)

    # get all original images
    # images = []
    # for img_path, _ in dataset.samples:
    #    print("get_embeddings(): img_path: " + img_path)
    #    img = cv2.imread(img_path)
    #    images.append(img)

    print("")
    print("")

    list_folder = next(os.walk(data_root))[1]
    list_nama = []
    try:
        for folder in list_folder:
            for i in range(len(os.listdir(data_root+folder))):
                list_nama.append(folder)
    except Exception as err:
        print(err)

    # print("get_embeddings(): list_nama " + str(list_nama))
    return embeddings, list_nama
