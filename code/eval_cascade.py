from process_img import *
from adaboost import *
from cascade import *
import numpy as np

def eval_cascade():
    # (options, args) = getopt.getopt(sys.argv[1:], '')
    # data_dir = args[0]
    pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, pos_train_imgs_int, \
    neg_train_imgs_int, pos_test_imgs_int, neg_test_imgs_int = get_img()

    # load adaboost classifier
    f = open('../output/cascade_classifier', 'rb')
    cas_classifier = pickle.load(f)
    f.close()

    acc, TP_rate, TN_rate, FP_rate, FN_rate, n_drops = cas_classifier.test_cas(pos_train_imgs_int, neg_train_imgs_int)
    emp = (1 - acc) * 100.0
    acc *= 100
    TP_rate *= 100.0
    FP_rate *= 100.0
    FN_rate *= 100.0

    f = open('../output/cascade_eval.txt', 'wt')
    print('Accuracy: %f%%' % acc, file=f)
    print('Detect Rate: %f%%' % TP_rate, file=f)
    print('Empirical Error: %f%%' % emp, file=f)
    print('False Positive : %f%%' % FP_rate, file=f)
    print('False Negative : %f%%' % FN_rate, file=f)
    print('', file=f)

    for i in range(len(n_drops)):
        print('Layer %d drops negative %d images' % (i, n_drops[i]), file=f)
    f.close()


if __name__ == "__main__":
    eval_cascade()

