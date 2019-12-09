import numpy as np
import sys
from haar_features import *
from typing import *
import random as rand

class Weak_Classifier(object):

    def __init__(self, feature: Haar_Feature, parity, theta):
        self.feature = feature
        self.parity = parity
        self.theta = theta
        self.alpha = 0.0
        self.accuracy = 0.0
        self.beta = 0.0
        self.weights = None
        self.diff = None
        self.last_weights = None

    def classify(self, img):
        if  self.parity * self.feature.get_feature_val(img) < self.parity * self.theta:
            return 1.0

        return 0.0


class Strong_Classifier(object):

    def __init__(self, weak_classifiers: List[Weak_Classifier], error_type):
        self.weak_classifiers = weak_classifiers
        self.error_type = error_type

    def classify(self, img):
        left = 0.0
        right = 0.0

        # find prediction of every weak classifiers
        for classifier in self.weak_classifiers:
            alpha = classifier.alpha
            predict = classifier.classify(img)
            left += alpha * predict
            right += alpha
        right *= 0.5

        # print('left ' + str(left))
        # print('right ' + str(right))

        if left >= right:
            return 1.0

        return 0.0

    # get emp, TP, TN, FP, FN rate on certain sets
    def get_rate(self, pos_imgs, neg_imgs):
        n_correct = 0.0
        TP = 0.0
        TN = 0.0
        FP = 0.0
        FN = 0.0

        n_pos = len(pos_imgs)
        n_neg = len(neg_imgs)
        n_img = n_pos + n_neg

        for img in pos_imgs:
            predict = self.classify(img)
            if predict == 1:
                n_correct += 1.0
                TP += 1.0
            elif predict == 0:
                FN += 1.0
            else:
                print('wrong')

        for img in neg_imgs:
            predict = self.classify(img)
            if predict == 0:
                n_correct += 1.0
                TN += 1.0
            elif predict == 1:
                FP += 1.0
            else:
                print('wrong')

        acc = (n_correct / n_img)
        TP_rate = (TP / n_pos)
        TN_rate = (TN / n_neg)
        FP_rate = (FP / n_neg)
        FN_rate = (FN / n_pos)

        return acc, TP_rate, TN_rate, FP_rate, FN_rate

# find best threshold and polarity for a feature
def get_best_params(feature: Haar_Feature, imgs, ys, weights):
    # sort features by value
    feature_vals = np.array([feature.get_feature_val(img) for img in imgs])
    indexes = np.argsort(feature_vals)
    feature_vals = feature_vals[indexes]
    ys = ys[indexes]
    weights = weights[indexes]

    # building running sum
    s_minus = 0.0
    s_plus = 0.0
    t_minus = 0.0
    t_plus = 0.0
    s_minuses = []
    s_pluses = []

    for y, weight in zip(ys, weights):
        if y < 0.5:
            s_minus += weight
            t_minus += weight
        else:
            s_plus += weight
            t_plus += weight
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)

    # find optimal threshold and parity
    min_error = float('inf')
    threshold = 0.0
    parity = 0.0
    for val, s_m, s_p in zip(feature_vals, s_minuses, s_pluses):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        if error_1 < min_error:
            min_error = error_1
            threshold = val
            parity = -1.0
        elif error_2 < min_error:
            min_error = error_2
            threshold = val
            parity = 1.0
    return parity, threshold

def get_error(predicts, ys, weights, error_type):
    error = 0.0
    if error_type == 'Empirical':
        error = (weights * abs(predicts - ys)).sum()

    elif error_type == 'False_Positive':
        for predict, y, weight in zip(predicts, ys, weights):
            if (predict == 1 and y == 0):
                error += weight

    elif error_type == 'False_Negative':
        for predict, y, weight in zip(predicts, ys, weights):
            if (predict == 0 and y == 1):
                error += weight

    if error == 0:
        error += 0.000001
    return error



def train_adaboost(pos_imgs, neg_imgs,  max_feature_width, max_feature_height, T , error_type = 'Empirical', min_T = 0, pre_train : Strong_Classifier = None):

    # generate features
    img_width = pos_imgs[0].shape[0] - 1 # original width = integral image width - 1
    img_height = pos_imgs[0].shape[1] - 1   # original height = integral image height - 1
    feature_list = get_feature_list(img_width, img_height, max_feature_width, max_feature_height)
    n_classifiers = len(feature_list)

    # initialize weights
    n_pos = len(pos_imgs)
    n_neg = len(neg_imgs)
    n_img = n_pos + n_neg
    weights_pos = [1.0 / (2 * n_pos)] * n_pos
    weights_neg = [1.0 / (2 * n_neg)] * n_neg
    weights = np.array(weights_pos + weights_neg)

    # Iterate T times
    imgs = np.concatenate((pos_imgs, neg_imgs), axis = 0)
    ys = np.array([1.0] * n_pos + [0.0] * n_neg)
    opt_classifiers = []
    if pre_train != None:
        opt_classifiers = pre_train.weak_classifiers
    prob_thre = 0.25
    print('start iterate')
    for t in range(min_T, T):
        # normalize the weights
        if len(opt_classifiers) > 0:
            weights = opt_classifiers[-1].last_weights
        weights = weights / weights.sum()

        # train weak classifier
        # get classifier list
        classifier_list = []
        i = 0
        for feature in feature_list:
            i += 1
            if rand.random() < prob_thre:
                continue
            parity, theta = get_best_params(feature, imgs, ys, weights)
            classifier = Weak_Classifier(feature, parity, theta)
            classifier_list.append(classifier)
            if (i % 100 == 0):
                sys.stdout.write("\r round: %d, find feature %d / %d" % (t, i, n_classifiers))
                sys.stdout.flush()
        print()
        print('round: %d - finishing finding classifiers' % (t))

        # find error for each classifier and get the optimal classifier
        opt_error = float('inf')
        opt_classifier = None
        opt_diffs = np.ones(n_img)
        i = 0
        for classifier in classifier_list:
            i += 1
            # get prediction all images
            predicts = []
            for img in imgs:
                predicts.append(classifier.classify(img))
            predicts = np.array(predicts)

            # find error and update optimal classifier
            diffs = abs(predicts - ys)
            error = get_error(predicts, ys, weights, error_type)
            # error = (weights * diffs).sum()
            if error < opt_error:
                opt_error = error
                opt_classifier = classifier
                opt_diffs = diffs

            if (i % 100 == 0):
                sys.stdout.write("\r round: %d, train classifier %d / %d" % (t, i, n_classifiers))
                sys.stdout.flush()

        # update weight
        opt_classifier.weights = weights
        beta_val = opt_error / (1.0 - opt_error)
        beta = np.full(n_img, beta_val)
        weights = weights * np.power(beta, np.ones(n_img) - opt_diffs)
        alpha = math.log(1.0 / beta_val, 2)

        # update classifier list
        opt_classifier.last_weights = weights
        opt_classifier.accuracy = 1.0 - opt_diffs.sum() / opt_diffs.shape[0]
        opt_classifier.alpha = alpha
        opt_classifier.beta = beta_val
        opt_classifier.diff = opt_diffs
        opt_classifiers.append(opt_classifier)
        print()
        print('round %d: feature accuracy: %f' % (t, opt_classifier.accuracy))

    strong_classifier = Strong_Classifier(opt_classifiers, error_type)
    return strong_classifier

