import cv2
import numpy as np
import tensorflow as tf


def rotate_image(img, angle):
    (rows, cols) = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(img, M, (cols, rows))


def load_img(path, img_size):
    img_size = (img_size, img_size)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    angle = np.random.randint(0, 360)
    img = rotate_image(img, angle)
    img = cv2.blur(img, (5, 5))
    img = cv2.resize(img, img_size)

    return img


def model_predictions(path):
    image_size = 64
    model_path = "conv_model/model.ckpt"
    image = np.array(load_img(path, image_size))
    image = np.array(image).reshape(-1, image_size, image_size, 1)
    y_true = np.array([1, 1]).reshape(-1, 2)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(model_path + '.meta')
        saver.restore(sess, model_path)
        # access a variable from the saved Graph, and so on:

        graph = tf.get_default_graph()
        accuracy = graph.get_tensor_by_name('accuracy:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        x = graph.get_tensor_by_name('x_placeholder:0')
        y = graph.get_tensor_by_name('y_placeholder:0')
        prediction = graph.get_tensor_by_name('predictions:0')
        correct_prediction = graph.get_tensor_by_name('correct_prediction:0')

        cp, a, o = sess.run([correct_prediction, accuracy, prediction],
                            feed_dict={x: image, y: y_true, keep_prob: 1.0})

    return o


path = 'lab.jpg'
prediction = model_predictions(path)

if prediction[0]:
    print('This is a Hotdog!')
else:
    print('This is not a Hotdog')
