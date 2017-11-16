import time

import numpy as np
import tensorflow as tf
from numpy.random import seed
from sklearn.model_selection import train_test_split

seed(1)
from tensorflow import set_random_seed

set_random_seed(2)

# Reset Graph
tf.reset_default_graph()


def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.05, name=name)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.05, shape=[shape])
    return tf.Variable(initial)


def create_convolutional_layer(input,
                               num_input_channels,
                               conv_filter_size,
                               num_filters):
    # We shall define the weights that will be trained using create_weights function.
    weights = weight_variable(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters],
                              name='conv_layer')
    # We create biases using the create_biases function. These are also trained.
    biases = bias_variable(num_filters)

    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    layer += biases

    # We shall be using max-pooling.
    layer = tf.nn.max_pool(value=layer,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME')
    # Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.relu(layer)

    return layer


def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,
                    num_inputs,
                    num_outputs,
                    use_relu=True):
    # Let's define trainable weights and biases.
    weights = weight_variable(shape=[num_inputs, num_outputs], name='fully_connected_layer')
    biases = bias_variable(num_outputs)

    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
#     acc = session.run(accuracy, feed_dict=feed_dict_train)
#     val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
#     msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
#     print(msg.format(epoch + 1, acc, val_acc, val_loss))


def run(X_, Y_, epochs=10, learning_rate=0.01, image_size=28, num_classes=2):
    start = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('Starting Convolutional Neural Network')
    nn_start = time.time()

    num_channels = 1

    X_ = np.array(X_).reshape(-1, image_size, image_size, num_channels)

    # Get data and TTS
    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    # Config



    ##Network graph params
    filter_size_conv1 = 3
    num_filters_conv1 = 32

    filter_size_conv2 = 3
    num_filters_conv2 = 32

    filter_size_conv3 = 3
    num_filters_conv3 = 64

    fc_layer_size = 128

    model_path = "conv_model/model.ckpt"
    image_size_sq = image_size * image_size
    batch_size = 50
    logs_path = "logs"
    training_epochs = epochs
    initial_learning_rate = learning_rate
    n_classes = 2
    input_dim = [None, image_size_sq]

    # global step
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)

    # learning rate policy
    decay_steps = int(len(X_train) / batch_size)
    learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                               global_step,
                                               decay_steps,
                                               decay_rate=0.96,
                                               staircase=True,
                                               name='exponential_decay_learning_rate')

    x = tf.placeholder(tf.float32, shape=[None, image_size, image_size, 1], name='x')

    y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    y_true_cls = tf.cast(y_true_cls, tf.float32)

    layer_conv1 = create_convolutional_layer(input=x,
                                             num_input_channels=num_channels,
                                             conv_filter_size=filter_size_conv1,
                                             num_filters=num_filters_conv1)

    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                             num_input_channels=num_filters_conv1,
                                             conv_filter_size=filter_size_conv2,
                                             num_filters=num_filters_conv2)

    layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                             num_input_channels=num_filters_conv2,
                                             conv_filter_size=filter_size_conv3,
                                             num_filters=num_filters_conv3)

    layer_flat = create_flatten_layer(layer_conv3)

    layer_fc1 = create_fc_layer(input=layer_flat,
                                num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                                num_outputs=fc_layer_size,
                                use_relu=True)

    layer_fc2 = create_fc_layer(input=layer_fc1,
                                num_inputs=fc_layer_size,
                                num_outputs=num_classes,
                                use_relu=False)

    y_pred = tf.nn.softmax(layer_fc2, name="y_pred")

    y_pred_cls = tf.argmax(y_pred, dimension=1)
    y_pred_cls = tf.cast(y_pred_cls, tf.float32)

    # Cost function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=layer_fc2, name='loss'))
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
    correct_prediction = tf.equal(y_pred_cls, y_true_cls, name='correct_prediction')

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

            accuracy = accuracy.eval(feed_dict={x: X_test, y_true: y_test})

            val_cost = sess.run(validation_cost, feed_dict={x: X_test, y_true: y_test})

            for step in range(batch_count):
                randidx = np.random.randint(len(X_train), size=batch_size)

                batch_x = np.array(X_train)[randidx]
                batch_y = np.array(y_train)[randidx]

                summary, c, training_step, lr = sess.run([summary_op, loss, global_step, learning_rate],
                                                         feed_dict={x: batch_x, y_true: batch_y})

                total_loss += c
                if step % batch_count == 0:
                    train_accuracy = accuracy.eval(feed_dict={x: batch_x, y_true: batch_y})
                    print("step %d, training accuracy %g" % (step, train_accuracy))

                    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
                    print(msg.format(epoch + 1, train_accuracy, accuracy, val_cost))

                optimizer.run(feed_dict={x: batch_x, y_true: batch_y})

                # write log
                writer.add_summary(summary, global_step=step)

            print('~~~~~~~~~~~~~~~~~~~~\n')

            print('Current Learning Rate: ', lr)

            print("Test accuracy %g" % accuracy)

            print("Validation Loss:", )

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
