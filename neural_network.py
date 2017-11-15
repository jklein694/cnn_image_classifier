import numpy as np
import tensorflow as tf


def get_data():
    from sklearn.model_selection import train_test_split

    X_
    Y_

    print(X_.shape)
    print(Y_.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    print('Images Read')

    return X_train, X_test, y_train, y_test



def build_graph(epochs=10, learning_rate=0.01, num_nodes_hl1=500, num_nodes_hl2=500, num_nodes_hl3=500, num_nodes_hl4=10):
    tf.reset_default_graph()

    num_nodes_hl1 = num_nodes_hl1
    num_nodes_hl2 = num_nodes_hl2
    num_nodes_hl3 = num_nodes_hl3
    num_nodes_hl4 = num_nodes_hl4

    training_epochs = epochs

    n_classes = 2
    learning_rate = learning_rate
    # The size of our imput data is 784
    input_dim = 784

    # None -> batch size can be any size, 784 -> flattened mnist image
    x = tf.placeholder('float', shape=[None, input_dim], name='x_placeholder')
    # target 10 output classes
    y = tf.placeholder('float', shape=[None, n_classes], name='y_placeholder')

    W1 = tf.Variable(tf.random_normal([input_dim, num_nodes_hl1], name='hidden_layer_1'))
    W2 = tf.Variable(tf.random_normal([num_nodes_hl1, num_nodes_hl2], name='hidden_layer_2'))
    W3 = tf.Variable(tf.random_normal([num_nodes_hl2, num_nodes_hl3], name='hidden_layer_3'))
    W4 = tf.Variable(tf.random_normal([num_nodes_hl3, num_nodes_hl4], name='hidden_layer_3'))
    out_w = tf.Variable(tf.random_normal([num_nodes_hl4, n_classes], name='hidden_layer_1'))

    # bias

    b1 = tf.Variable(tf.random_normal([num_nodes_hl1]))
    b2 = tf.Variable(tf.random_normal([num_nodes_hl2]))
    b3 = tf.Variable(tf.random_normal([num_nodes_hl3]))
    b4 = tf.Variable(tf.random_normal([num_nodes_hl4]))
    out_b = tf.Variable(tf.random_normal([n_classes]))

    # y is our prediction
    l1 = tf.add(tf.matmul(x, W1), b1)

    # Add activation function to run this data
    l1 = tf.nn.relu(l1, name='relu_l1')

    l2 = tf.add(tf.matmul(l1, W2), b2)
    l2 = tf.nn.relu(l2, name='relu_l2')

    l3 = tf.add(tf.matmul(l2, W3), b3)
    l3 = tf.nn.relu(l3, name='relu_l3')

    l4 = tf.add(tf.matmul(l3, W4), b4)
    l4 = tf.nn.relu(l4, name='relu_l3')

    output = tf.add(tf.matmul(l4, out_w), out_b, name='output')

    # specify cost function

    # this is our cost
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=output, name='loss'))
    tf.summary.scalar("cost", cost)

    validation_cost = cost
    tf.summary.scalar("validation_cost", validation_cost)


    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, name='optimizer')

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1), name='correct_prediction')
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
    tf.summary.scalar("accuracy", accuracy)

    # create a summary for our cost and accuracy



    # merge all summaries into a single "operation" which we can execute in a session
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver(max_to_keep=1)

    init = tf.global_variables_initializer()

    print('Graph Built')

    return x, y, cost, validation_cost, accuracy, optimizer, summary_op, saver, init, training_epochs


def execute_nn(X_train, X_test, y_train, y_test, x, y, cost, validation_cost, accuracy, optimizer, summary_op, saver, init,
               training_epochs):

    # Need to create the "save_model" directory first
    model_path = "saved_model/model.ckpt"
    # These are the number of nodes we want to have in each layer,
    # we can set these values to whatever we want.

    # config
    batch_size = 100
    training_epochs = training_epochs
    logs_path = "logs"

    with tf.Session() as sess:
        # variables need to be initialized before we can use them

        sess.run(init)

        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph(), max_queue=2)

        # perform training cycles
        for epoch in range(training_epochs):

            batch_count = int(len(X_train) / batch_size)
            total_loss = 0
            for i in range(int(len(X_train) / batch_size)):
                randidx = np.random.randint(len(X_train), size=batch_size)

                batch_x = np.array(X_train)[randidx]
                batch_y = np.array(y_train)[randidx]

                # perform the operations we defined earlier on batch
                _, summary, c = sess.run([optimizer, summary_op, cost], feed_dict={x: batch_x, y: batch_y})

                total_loss += c
                # write log
                writer.add_summary(summary, epoch * batch_count + i)



            print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: X_test, y: y_test}))

            print("Validation Loss:", sess.run(validation_cost, feed_dict={x: X_test, y: y_test}))

            print('Epoch ', epoch, ' completed out of ', training_epochs, ', loss: ', total_loss)

        # Save model weights to disk

        save_path = saver.save(sess, model_path)

        print("Model saved in file: %s" % save_path)

        return total_loss, accuracy

# def clean_logs():
#     import os
#     if len(os.listdir('logs'))> 3:
#         print()
#             os.remove()



def main():
    X_train, X_test, y_train, y_test = get_data()
    x, y, cost, validation_cost, accuracy, optimizer, summary_op,\
    saver, init, training_epochs = build_graph(epochs=10, learning_rate=.0001)

    execute_nn(X_train, X_test, y_train, y_test, x, y, cost,
               validation_cost, accuracy, optimizer, summary_op,
               saver, init, training_epochs)


if __name__ == "__main__":
    main()
