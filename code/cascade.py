import numpy as np
from adaboost import *
from haar_features import *
from collections import defaultdict
import pickle

class Cascade_Classifier(object):

    def __init__(self, ada_classifiers: List[Strong_Classifier]):
        self.ada_classifiers = ada_classifiers

    def classifiy(self, img):
        for ada_classifier in self.ada_classifiers:
            if ada_classifier.classify(img) == 0:
                return 0.0

        return 1.0

    def test_cas(self, pos_imgs, neg_imgs):
        n_drops = defaultdict(int)
        n_pos = len(pos_imgs)
        n_neg = len(neg_imgs)
        n_img = n_pos + n_neg
        n_correct = 0.0

        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0

        for img in pos_imgs:
            i = 0
            predict = 1
            for ada_classifier in self.ada_classifiers:
                if ada_classifier.classify(img) == 0:
                    n_drops[i] += 1
                    predict = 0
                    break
                i += 1

            if predict == 1:
                n_correct += 1.0
                TP += 1.0
            elif predict == 0:
                FN += 1.0

        for img in neg_imgs:
            i = 0
            predict = 1
            for ada_classifier in self.ada_classifiers:
                if ada_classifier.classify(img) == 0:
                    n_drops[i] += 1
                    predict = 0
                    break
                i += 1

            if predict == 0:
                n_correct += 1.0
                TN += 1.0
            elif predict == 1:
                FP += 1.0

        acc = n_correct / n_img
        TP_rate = TP / n_pos
        TN_rate = TN / n_neg
        FP_rate = FP / n_neg
        FN_rate = FN / n_pos
        return acc, TP_rate, TN_rate, FP_rate, FN_rate, n_drops

# move predicted neg imgs in pos_img set to neg_img set, and find accuracy
def move_to_negs(ada_classifier, pos_imgs, neg_imgs):
    i = 0
    pos_drop_idx = []
    neg_drop_idx = []

    for img in pos_imgs:
        predict = ada_classifier.classify(img)
        if predict == 0:
            pos_drop_idx.append(i)
        i += 1

    i = 0
    for img in neg_imgs:
        predict = ada_classifier.classify(img)
        if predict == 0:
            neg_drop_idx.append(i)
        i += 1
    new_pos_imgs = np.delete(pos_imgs, pos_drop_idx, axis = 0)
    new_neg_imgs = np.delete(neg_imgs, neg_drop_idx, axis = 0)

    return new_pos_imgs, new_neg_imgs

def train_cascade(pos_imgs, neg_imgs, max_feature_width, max_feature_height, n_layer):
    prev_detect_rate = float('inf')
    prev_FP_rate = float('inf')
    prev_T = 0


    opt_classifiers = []

    for l in range(n_layer):
        print('Cascade layer %d' % l)
        T = prev_T
        while(True):
            step = 1
            T += step
            if T > 40:
                break
            print('Cascade layer %d - ada rounds %d' % (l, T))
            if T > prev_T + step:
                ada_classifier = train_adaboost(pos_imgs, neg_imgs, max_feature_width, max_feature_height, T, min_T = T - step, pre_train=ada_classifier)
            else:
                ada_classifier = train_adaboost(pos_imgs, neg_imgs, max_feature_width, max_feature_height, T)
            acc, detect_rate, TN_rate, FP_rate, FN_rate = ada_classifier.get_rate(pos_imgs, neg_imgs)
            print('detect rate: %f, FP rate: %f' % (detect_rate, FP_rate))
            if (detect_rate < prev_detect_rate and FP_rate < prev_FP_rate):
                prev_detect_rate = detect_rate
                prev_FP_rate = FP_rate
                opt_classifiers.append(ada_classifier)
                prev_T = T

                # move predicted neg imgs in pos_img set to neg_img set
                pos_imgs, neg_imgs = move_to_negs(ada_classifier, pos_imgs, neg_imgs)

                break


    cascade_classifier = Cascade_Classifier(opt_classifiers)

    return cascade_classifier
