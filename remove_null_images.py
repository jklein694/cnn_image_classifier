import os

import cv2
import numpy as np


def remove_invalid(dir_paths):
    print('~~~~~~~~~~~~~~~~~~~\n')
    print('Removing Invalid Images')
    for dir_path in dir_paths:
        count = 0
        for img in os.listdir(dir_path):
            for invalid in os.listdir('invalid'):
                try:
                    current_image_path = str(dir_path) + '/' + str(img)
                    invalid = cv2.imread('invalid/' + str(invalid))
                    question = cv2.imread(current_image_path)
                    if invalid.shape == question.shape and not (np.bitwise_xor(invalid, question).any()):
                        count += 1
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))
        print('{} invalid images removed'.format(count))


paths = ['not_hotdog',
         'hotdog']
