import os
import time

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from sklearn.model_selection import train_test_split

start = time.time()


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1, name=name)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, padding):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding)


def max_pool_2x2(x, padding):
    #                             size of window      movement of window
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)


def max_pool_4x2(x, padding):
    #                             size of window      movement of window
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 2, 2, 1], padding=padding)


def max_pool_4x4(x, padding):
    #                             size of window       movement of window
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding=padding)


def run(X_, Y_, epochs=10, learning_rate=0.01, image_size=28):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('Starting Convolutional Neural Network')
    nn_start = time.time()

    X_ = np.array(X_).reshape(-1, 784)

    # Get data and TTS
    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    # Config
    model_path = "conv_model/model.ckpt"
    image_size_sq = image_size * image_size
    batch_size = 50
    logs_path = "logs"
    training_epochs = epochs
    initial_learning_rate = learning_rate
    n_classes = 2
    input_dim = [None, image_size_sq]

    # Reset Graph
    tf.reset_default_graph()

    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    # learning rate policy
    decay_steps = int(len(X_train) / batch_size)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate=0.92,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    # Weights
    w_conv1 = weight_variable([5, 5, 1, 32], name='hidden_layer_1')
    w_conv2 = weight_variable([5, 5, 32, 64], name='hidden_layer_2')
    w_conv3 = weight_variable([5, 5, 64, 128], name='hidden_layer_3')
    w_fc1 = weight_variable([800 * 128, 1024], name='fully_connected_layer')
    out_w = weight_variable([1024, n_classes], name='hidden_layer_out')

    # bias
    b_conv1 = bias_variable([32])
    b_conv2 = bias_variable([64])
    b_conv3 = bias_variable([128])
    b_fc1 = bias_variable([1024])
    out_b = bias_variable([n_classes])

    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Placeholders
    x = tf.placeholder('float', shape=input_dim, name='x_placeholder')
    x_img = tf.reshape(x, [-1, image_size, image_size, 1])

    y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')
    y = tf.reshape(y, [-1, 2])

    # Convolutional Layers
    h_conv1 = tf.nn.relu(conv2d(x_img, w_conv1, padding='SAME') + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1, padding='SAME')

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, padding='SAME') + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2, padding='SAME')

    h_conv3 = tf.nn.relu(conv2d(h_pool2, w_conv3, padding='SAME') + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3, padding='SAME')

    h_pool2_flat = tf.reshape(h_pool3, [-1, 800 * 128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    predictions = tf.add(tf.matmul(h_fc1_drop, out_w), out_b, name='predictions')

    # Cost function
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predictions, name='loss'))
    tf.summary.scalar("Training Loss", loss)

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss,
                                                               # aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE,
                                                               global_step=global_step,
                                                               name='optimizer')

    # Validation cost
    validation_cost = loss
    tf.summary.scalar("validation_cost", validation_cost)
    tf.summary.histogram('Hist Val Cost', validation_cost)

    # Correct Predictions and Accuracy
    correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1), name='correct_prediction')

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram('Hist Accuracy', accuracy)

    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    # Save only one model
    saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto(inter_op_parallelism_threads=44,
                            intra_op_parallelism_threads=44)

    init = tf.global_variables_initializer()

    graph_time = time.time() - nn_start

    if graph_time >= 60:
        graph_time = graph_time / 60
        print('Graph Built: {} minutes'.format(round(graph_time, 0)))
    else:
        print('Graph Built: {} seconds'.format(round(graph_time, 0)))

    with tf.Session(config=config) as sess:
        print('Session Started')
        print('--------------- \n')
        # variables need to be initialized before we can use them

        sess.run(init)

        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), max_queue=2)

        # perform training cycles
        for epoch in range(training_epochs):

            epoch_start = time.time()

            batch_count = int(len(X_train) / batch_size)
            print('Number of Batches: ', batch_count)
            total_loss = 0
            for step in range(batch_count):
                randidx = np.random.randint(len(X_train), size=batch_size)

                batch_x = np.array(X_train)[randidx]
                batch_y = np.array(y_train)[randidx]

                summary, c, training_step, lr = sess.run([summary_op, loss, global_step, learning_rate],
                                                         feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                total_loss += c
                if step % 50 == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
                    print("step %d, training accuracy %g" % (step, train_accuracy))

                optimizer.run(feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

                # write log
                writer.add_summary(summary, global_step=step)

            print('~~~~~~~~~~~~~~~~~~~~\n')

            print('Current Learning Rate: ', lr)

            print("Test accuracy %g" % accuracy.eval(feed_dict={x: X_test, y: y_test, keep_prob: 1.0}))

            print("Validation Loss:", sess.run(validation_cost, feed_dict={x: X_test, y: y_test, keep_prob: 0.5}))

            print('Epoch ', epoch + 1, ' completed out of ', training_epochs, ', loss: ', total_loss)

            epoch_time = time.time() - epoch_start
            if epoch_time >= 60:
                epoch_time = epoch_time / 60
                print('Epoch Time: {} minutes'.format(round(epoch_time, 0)))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
            else:
                print('Epoch Time: {} seconds'.format(round(epoch_time, 0)))
                print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

        # Save model
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

        total_time = time.time() - start
        if total_time >= 60:
            total_time = total_time / 60
            print('Epoch Time: {} minutes'.format(round(total_time, 0)))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')
        else:
            print('Epoch Time: {} seconds'.format(round(total_time, 0)))
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ \n')

        return total_loss, accuracy
