import numpy as np
import math
# from enum import Enum


'''
generate integral image
'''
def get_int_img(img):
    # find integral
    integral_img = np.cumsum(img, axis=0)
    integral_img = np.cumsum(integral_img, axis=1)

    # add boundary row and column
    integral_img = np.insert(integral_img, 0, values=np.zeros(integral_img.shape[1]), axis=0)
    integral_img = np.insert(integral_img, 0, values=np.zeros(integral_img.shape[0]), axis=1)
    return integral_img

'''
find rectangular size from integral image
'''
def get_rec_sum(integral_img, x, y, width, height):
    x = int(x)
    y = int(y)
    width = int(width)
    height = int(height)

    # if width == 0 and height == 0:
    #     return integral_img[y][x]

    # find four integral of four corners
    A = integral_img[y][x]
    B = integral_img[y + height][x]
    C = integral_img[y][x + width]
    D = integral_img[y + height][x + width]

    return D + A - B - C

'''
five types of feature
'''

# Parent class for all types of features
class Haar_Feature(object):

    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.name = ''

    # get feature value of a image
    def get_feature_val(self, img):
        pass

# two vertical feature
class Feature_2V(Haar_Feature):
    def __init__(self, x, y, width = 1, height = 2):
        super().__init__(x, y, width, height)
        self.name = '1 (two vertical)'

    def get_feature_val(self, img):
        grey = get_rec_sum(img, self.x, self.y, self.width, self.height / 2.0)
        white = get_rec_sum(img, self.x, self.y + (self.height / 2.0), self.width, self.height / 2.0)
        return white - grey

# two horizontal feature
class Feature_2H(Haar_Feature):
    def __init__(self, x, y, width = 2, height = 1):
        super().__init__(x, y, width, height)
        self.name = '2 (two horizontal)'

    def get_feature_val(self, img):
        white = get_rec_sum(img, self.x, self.y, self.width / 2.0, self.height)
        grey = get_rec_sum(img, self.x + (self.width / 2.0), self.y, self.width / 2.0, self.height)
        return white - grey

# three horizontal feature
class Feature_3H(Haar_Feature):
    def __init__(self, x, y, width = 3, height = 1):
        super().__init__(x, y, width, height)
        self.name = '3 (three horizontal)'


    def get_feature_val(self, img):
        white = get_rec_sum(img, self.x + (self.width / 3.0), self.y, self.width / 3.0, self.height)
        grey = get_rec_sum(img, self.x, self.y, self.width / 3.0, self.height) \
                + get_rec_sum(img, self.x + (2 * self.width / 3.0), self.y, self.width / 3.0, self.height)
        return white - grey

# three vertical feature
class Feature_3V(Haar_Feature):
    def __init__(self, x, y, width = 1, height = 3):
        super().__init__(x, y, width, height)
        self.name = '4 (three vertical)'


    def get_feature_val(self, img):
        white = get_rec_sum(img, self.x, self.y + + (self.height / 3.0), self.width, self.height / 3.0)
        grey = get_rec_sum(img, self.x, self.y, self.width, self.height / 3.0) \
                + get_rec_sum(img, self.x , self.y + (2 * self.height / 3.0), self.width, self.height / 3.0)
        return white - grey

# four feature
class Feature_4(Haar_Feature):
    def __init__(self, x, y, width = 2, height = 2):
        super().__init__(x, y, width, height)
        self.name = '5 (four)'


    def get_feature_val(self, img):
        white = get_rec_sum(img, self.x, self.y, self.width / 2.0, self.height / 2.0) \
                + get_rec_sum(img, self.x + (self.width / 2.0), self.y + (self.height / 2.0), self.width / 2.0, self.height / 2.0)
        grey = get_rec_sum(img, self.x + (self.width / 2.0), self.y, self.width / 2.0, self.height / 2.0)\
               + get_rec_sum(img, self.x, self.y + (self.height / 2.0), self.width / 2.0, self.height / 2.0)
        return white - grey

'''
generate all possible features for given image size and max feature size
'''
def get_feature_list(img_width, img_height, max_feature_width, max_feature_height):
    Feature_Classes = [Feature_2V, Feature_2H, Feature_3H, Feature_3V, Feature_4]
    feature_list = []

    f = open('../output/n_features.txt', 'wt')


    for Feature_Class in Feature_Classes:
        n_feature_type = 0  # number of features of certain type
        feature_base = Feature_Class(0, 0)  # generate a simple feature to get basic feature size and name
        unit_width = feature_base.width
        unit_height = feature_base.height
        feature_name = feature_base.name

        # different feature size with unit size of (1, 2), (2, 1), (1, 3) etc.
        for feature_width in range(unit_width, max_feature_width + 1, unit_width):
            for feature_height in range(unit_height, max_feature_height + 1, unit_height):

                # different possible positions (indexed by top left corner)
                for x in range(img_width - feature_width + 1):
                    for y in range(img_height - feature_height + 1):
                        feature = Feature_Class(x, y, feature_width, feature_height)
                        feature_list.append(feature)
                        n_feature_type += 1
        print('There are %d type %s features.' % (n_feature_type, feature_name), file = f)

    print('The total number of Haar Features is: %d.' % (len(feature_list)), file = f)
    f.close()
    return feature_list

