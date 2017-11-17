import cv2
import numpy as np
import tensorflow as tf


def input_parser(filenames, image_size):
    tests = []

    for file_path in filenames:
        images = []
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # Resizing the image to our desired size and
        # preprocessing will be done exactly as done during training
        image = cv2.resize(image, (image_size, image_size))
        images.append(image)
        images = np.array(images, dtype=np.uint8)
        images = images.astype('float32')
        X_check = images.reshape(-1, image_size, image_size, 1)

        tests.append(X_check)

    return X_check


def restore_model(tests, model_path, image_size):
    model_path = model_path

    y_correct = np.array([0.00000000, 1.000000000]).reshape(-1, 2)

    y_hats = []



    for test in tests:

        test = np.array(test).reshape(-1, image_size, image_size, 1)

        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(model_path + '.meta')
            saver.restore(sess, model_path)

            graph = tf.get_default_graph()
            keep_prob = graph.get_tensor_by_name('keep_prob:0')
            x = graph.get_tensor_by_name('x_placeholder:0')
            y = graph.get_tensor_by_name('y_placeholder:0')

            correct_prediction = graph.get_tensor_by_name('correct_prediction:0')

            y_hat = sess.run(correct_prediction, feed_dict={x: test, y: y_correct, keep_prob: 1.0})

            y_hats.append(y_hat)

        return y_hats


def run(model_path, image_size):
    model_path = model_path
    file_path = ['hotdog.jpg', 'not_hotdog.jpg']
    tests = input_parser(file_path, image_size)
    prediction = restore_model(tests, model_path, image_size)

    for i in range(len(prediction)):
        print('Hotdog test')
        print('------------ \n')
        if prediction[i][0]:
            guess = 'Hotdog'
            print('This is a ', guess, '! My CNN was right.')
        else:
            guess = 'not a hotdog'
            print('This is ', guess, ' and my CNN was wrong.')
        print('\t')
        print('\n')

        print('Not hotdog test')
        print('--------------- \n')

        if prediction[i][0] == True:
            guess = 'Hotdog'
            print('This is a ', guess, ' :( My CNN was wrong.')
        else:
            guess = 'not a hotdog'
            print('This is ', guess, ' and my CNN was right!.')
        print('\n \n')
