from haar_features import *
from process_img import *
from adaboost import *
import pickle

# draw images with feature
def draw_img(org_img, classifier: Weak_Classifier):
    feature = classifier.feature
    white_val = 0.0 if classifier.parity == 1 else 1.0
    grey_val = 1.0 if classifier.parity == 1 else 0.0
    x = feature.x
    y = feature.y
    width = feature.width
    height = feature.height
    img = np.copy(org_img)

    # Image.fromarray(np.uint8(img * 255.)).resize((100, 100)).show()

    if feature.name == '1 (two vertical)':
        img[int(y): int(y + height / 2), int(x): int(x + width)] = white_val
        img[int(y + height / 2): int(y + height), int(x): int(x + width)] = grey_val

    elif feature.name == '2 (two horizontal)':
        img[int(y): int(y + height), int(x): int(x + width / 2)] = white_val
        img[int(y): int(y + height), int(x + width / 2): int(x + width)] = grey_val

    elif feature.name == '3 (three horizontal)':
        img[int(y): int(y + height), int(x): int(x + width / 3)] = grey_val
        img[int(y): int(y + height), int(x + width / 3): int(x + 2 * width / 3)] = white_val
        img[int(y): int(y + height), int(x + 2 * width / 3): int(x + width)] = grey_val

    elif feature.name == '4 (three vertical)':
        img[int(y): int(y + height / 3), int(x): int(x + width)] = grey_val
        img[int(y + height / 3): int(y + 2 * height / 3), int(x): int(x + width)] = white_val
        img[int(y + 2 * height / 3): int(y + height), int(x): int(x + width)] = white_val

    elif feature.name == '5 (four)':
        img[int(y): int(y + height / 2), int(x): int(x + width / 2)] = white_val
        img[int(y + height / 2): int(y + height), int(x): int(x + width / 2)] = grey_val
        img[int(y): int(y + height / 2), int(x + width / 2): int(x + width)] = grey_val
        img[int(y + height / 2): int(y + height), int(x + width / 2): int(x + width)] = white_val

    img_show = Image.fromarray(np.uint8(img * 255.)).resize((500, 500))
    return img_show

# evaluate features in classifier
def eval_features(strong_classifer: Strong_Classifier, img, pos_test_imgs_int, neg_test_imgs_int):
    # if origin
    weak_classifiers = strong_classifer.weak_classifiers
    i = 0
    check_idx = [1, 3, 5, 10]
    check_classifiers_list = [weak_classifiers[:index] for index in check_idx]

    f = open('../output/feature_eval.txt', 'wt')

    for classifiers in check_classifiers_list:
        # print(len(classifiers))
        high_acc = 0.0
        high_classifier = None

        print('Adaboost Round: %d' % check_idx[i], file = f)
        print('', file = f)

        j = 0
        for classifier in classifiers:
            j += 1
            print('Feature number %d:' % (j), file = f)
            feature = classifier.feature
            print('Type: %s' % feature.name[3:-1], file = f)
            print('Position: (%d, %d)' % (feature.x, feature.y), file = f)
            print('Width: %d' % feature.width, file = f)
            print('Height: %d' % feature.height, file = f)
            print('Threshold: %f' % classifier.theta, file = f)
            print('Training accuracy: %f' % classifier.accuracy, file = f)
            print('', file = f)

        print('==================================', file = f)
        print('', file = f)

        # draw top feature
        for classifier in classifiers:
            # high_classifier = classifiers[-1]
            # high_acc = high_classifier.accuracy
            if classifier.accuracy > high_acc:
                high_acc = classifier.accuracy
                high_classifier = classifier

        img_show = draw_img(img, high_classifier)
        name = '../output/img_feature_' + str(check_idx[i]) + '.png'
        img_show.save(name)

        i += 1

    f.close()

    f = open('../output/adaboost_eval_1.txt', 'wt')

    i = 0
    for classifiers in check_classifiers_list:
        classifier = Strong_Classifier(classifiers, 'Empirical')
        acc, TP_rate, TN_rate, FP_rate, FN_rate = classifier.get_rate(pos_test_imgs_int, neg_test_imgs_int)
        acc *= 100.0
        FP_rate *= 100.0
        FN_rate *= 100.0

        # write into file
        print('Adaboost Round: %d' % check_idx[i], file = f)
        print('Total Accuracy: %f%%' % acc, file=f)
        print('False Positive : %f%%' % FP_rate, file=f)
        print('False Negative : %f%%' % FN_rate, file=f)
        print('', file=f)
        i += 1
    f.close()

def eval_ada_classifier(classifiers: List[Strong_Classifier], pos_test_imgs_int, neg_test_imgs_int):

    f = open('../output/adaboost_eval_2.txt', 'wt')

    for classifier in classifiers:
        acc, TP_rate, TN_rate, FP_rate, FN_rate = classifier.get_rate(pos_test_imgs_int, neg_test_imgs_int)
        acc *= 100.0
        FP_rate *= 100.0
        FN_rate *= 100.0

        # write into file
        print('Criterion: %s' % classifier.error_type, file=f)
        print('Total Accuracy: %f%%' % acc, file=f)
        print('False Positive : %f%%' % FP_rate, file=f)
        print('False Negative : %f%%' % FN_rate, file=f)
        print('', file=f)

    f.close()


def eval_adaboost():
    # (options, args) = getopt.getopt(sys.argv[1:], '')
    # data_dir = args[0]
    pos_train_imgs, neg_train_imgs, pos_test_imgs, neg_test_imgs, pos_train_imgs_int, \
    neg_train_imgs_int, pos_test_imgs_int, neg_test_imgs_int = get_img()

    # load adaboost classifier
    f = open('../output/adaboost_classifier_10', 'rb')
    classifier = pickle.load(f)
    f.close()

    # evaluate features in classifier
    eval_features(classifier, pos_test_imgs[2],pos_test_imgs_int, neg_test_imgs_int)


    # evaluate adaboost classifier with different error types
    # read classifiers from file
    classifiers = []

    f = open('../output/adaboost_classifier_Empirical', 'rb')
    cla = pickle.load(f)
    classifiers.append(cla)
    f.close()

    f = open('../output/adaboost_classifier_False_Positive', 'rb')
    cla = pickle.load(f)
    classifiers.append(cla)
    f.close()

    f = open('../output/adaboost_classifier_False_Negative', 'rb')
    cla = pickle.load(f)
    classifiers.append(cla)
    f.close()

    eval_ada_classifier(classifiers, pos_test_imgs_int, neg_test_imgs_int)

if __name__ == "__main__":
    eval_adaboost()