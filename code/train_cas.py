import numpy as np
import pickle
from cascade import *
from process_img import *

def train_cas():
    pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, pos_train_imgs_int, \
    neg_train_imgs_int, pos_test_imgs_int, neg_test_imgs_int = get_img()

    max_feature_width = 8
    max_feature_height = 8

    classifier = train_cascade(pos_train_imgs_int, neg_train_imgs_int, max_feature_width, max_feature_height, n_layer=3)

    data = classifier
    f = open('../output/cascade_classifier', 'wb')
    pickle.dump(data, f)
    f.close()

if __name__ == "__main__":
    train_cas()
