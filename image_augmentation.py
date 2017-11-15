import glob

import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def rotate_image(img, angle):
    (rows, cols) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def load_rotate__blur_img(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    angle = np.random.randint(0, 360)
    img = rotate_image(img, angle)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, img_size)
    return img


def load_rotate(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    angle = np.random.randint(0, 360)
    img = rotate_image(img, angle)
    img = cv2.resize(img, img_size)
    return img


def load_blur_img(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, img_size)
    return img


def load_img(path, img_size):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, img_size)
    return img


def load_img_class(class_path, class_label, class_size, img_size):
    x = []
    y = []

    #
    for path in class_path:
        img = np.array(load_img(path, img_size))
        x.append(img)
        y.append(class_label)

    while len(x) < class_size / 4:
        rand_idx = np.random.randint(0, len(class_path))
        img = np.array(load_rotate(class_path[rand_idx], img_size))
        x.append(img)
        y.append(class_label)

    while len(x) < class_size / 2:
        rand_idx = np.random.randint(0, len(class_path))
        img = np.array(load_blur_img(class_path[rand_idx], img_size))
        x.append(img)
        y.append(class_label)

    while len(x) < class_size * (3 / 4):
        rand_idx = np.random.randint(0, len(class_path))
        img = np.array(load_rotate__blur_img(class_path[rand_idx], img_size))
        x.append(img)
        y.append(class_label)

    while len(x) < class_size:
        rand_idx = np.random.randint(0, len(class_path))
        img = np.array(load_rotate__blur_img(class_path[rand_idx], img_size))
        x.append(img)
        y.append(class_label)

    return x, y


def load_data(img_size, class_size):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('Starting Image Augmentation')

    hotdogs = glob.glob('hotdog/**/*.jpg', recursive=True)

    not_hotdogs = glob.glob('not_hotdog/**/*.jpg', recursive=True)

    thresh_size = img_size

    img_size = (img_size, img_size)

    x_hotdog, y_hotdog = load_img_class(hotdogs, 0, class_size, img_size)
    print('Hotdogs Images Augmented')

    x_not_hotdog, y_not_hotdog = load_img_class(not_hotdogs, 1, class_size, img_size)
    print('Not Hotdogs Images Augmented')

    print("There are", len(x_hotdog), "hotdog images")
    print("There are", len(x_not_hotdog), "not hotdog images")

    X = np.array(x_hotdog + x_not_hotdog)
    y = np.array(y_hotdog + y_not_hotdog)

    y = np.array(y).reshape(-1, 1)

    encode = OneHotEncoder()
    encode.fit(y)
    y = encode.transform(y).toarray()

    X = (X - X.min(0)) / X.ptp(0)

    # if thresh_size <= 56:
    #     np.savetxt('X.csv', X, delimiter=',')
    #     np.savetxt('y.csv', y, delimiter=',')

    print('x tensor shape: {}'.format(np.array(X).shape))
    print('y tensor shape: {}'.format(np.array(y).shape))

    return X, y
