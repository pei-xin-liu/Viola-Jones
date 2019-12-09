import numpy as np
from haar_features import *
from adaboost import *
from eval_adaboost import *
from process_img import *
from PIL import Image
import os
import getopt
import sys
import pickle

def train_ada():
    # (options, args) = getopt.getopt(sys.argv[1:], '')
    # data_dir = args[0]
    pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, pos_train_imgs_int, \
    neg_train_imgs_int, pos_test_imgs_int, neg_test_imgs_int = get_img()

    max_feature_width = 8
    max_feature_height = 8

    classifier = train_adaboost(pos_train_imgs_int, neg_train_imgs_int, max_feature_width, max_feature_height, T=10)

    data = classifier
    f = open('../output/adaboost_classifier_10', 'wb')
    pickle.dump(data, f)
    f.close()

def train_ada_error():
    pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, pos_train_imgs_int, \
    neg_train_imgs_int, pos_test_imgs_int, neg_test_imgs_int = get_img()

    max_feature_width = 8
    max_feature_height = 8

    classifier_emp = train_adaboost(pos_train_imgs_int, neg_train_imgs_int, max_feature_width, max_feature_height,
                                T=5, error_type='Empirical')

    data = classifier_emp
    name = '../output/adaboost_classifier_' + str(data.error_type)
    f = open(name, 'wb')
    pickle.dump(data, f)
    f.close()

    classifier_FP = train_adaboost(pos_train_imgs_int, neg_train_imgs_int, max_feature_width, max_feature_height,
                                T=5, error_type='False_Positive')

    data = classifier_FP
    name = '../output/adaboost_classifier_' + str(data.error_type)
    f = open(name, 'wb')
    pickle.dump(data, f)
    f.close()

    classifier_FN = train_adaboost(pos_train_imgs_int, neg_train_imgs_int, max_feature_width, max_feature_height,
                                T=5, error_type='False_Negative')
    data = classifier_FN
    name = '../output/adaboost_classifier_' + str(data.error_type)
    f = open(name, 'wb')
    pickle.dump(data, f)
    f.close()


if __name__ == "__main__":
    train_ada()
    train_ada_error()
