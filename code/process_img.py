import numpy as np
from haar_features import *
from adaboost import *
from eval_adaboost import *
from PIL import Image
import os
import getopt
import sys
import pickle

def read_img(path):
    imgs = []
    for file in os.listdir(path):
        img = np.array(Image.open((os.path.join(path, file))), dtype=np.float64)
        # img /= img.max()
        img = np.array(img).astype(np.float32) / 255.
        imgs.append(img)
    return imgs

def get_img():
    data_dir = '../dataset'
    pos_train_dir = '%s/trainset/faces/' % data_dir
    neg_train_dir = '%s/trainset/non-faces/' % data_dir
    pos_test_dir = '%s/testset/faces/' % data_dir
    neg_test_dir = '%s/testset/non-faces/' % data_dir

    pos_train_imgs = read_img(pos_train_dir)
    neg_train_imgs = read_img(neg_train_dir)
    pos_test_imgs = read_img(pos_test_dir)
    neg_test_imgs = read_img(neg_test_dir)

    # get integral image
    pos_train_imgs_int = np.array([get_int_img(img) for img in pos_train_imgs])
    neg_train_imgs_int = np.array([get_int_img(img) for img in neg_train_imgs])
    pos_test_imgs_int = np.array([get_int_img(img) for img in pos_test_imgs])
    neg_test_imgs_int = np.array([get_int_img(img) for img in neg_test_imgs])

    return pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, pos_train_imgs_int, neg_train_imgs_int, pos_test_imgs_int, neg_test_imgs_int
