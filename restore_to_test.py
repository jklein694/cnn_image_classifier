import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


def model_predictions(X_, Y_, model_path, image_size):
    model_path = model_path

    X_ = np.array(X_).reshape(-1, image_size * image_size)

    X_train, X_test, y_train, y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)

    predictions = []

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

        cp, a, o = sess.run([correct_prediction, accuracy, tf.argmax(prediction, 1)],
                            feed_dict={x: X_test, y: y_test, keep_prob: 0.1})

        predictions.append(o)

    actuals = [y[1] for y in y_test]
    actuals = np.array(actuals)

    predictions = np.array(predictions).reshape(6000, )

    print(predictions.shape)
    print(actuals.shape)

    cnf_matrix = confusion_matrix(actuals, predictions)

    return predictions, actuals, cnf_matrix


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=' ')
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=' ')
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=' ')
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=' ')
        print()


def plot_confusion_matrix(cm, y_test, y_pred, class_names,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('\n')
        print("Normalized confusion matrix")
    else:
        print('\n')
        print('Confusion matrix, without normalization')
    print_cm(cm, class_names)
    text_labels = [['True Negative', 'False Positive'],
                   ['False Negative', 'True Positive']]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i - 0.1, format(cm[i, j], fmt),
                 verticalalignment='bottom',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        plt.text(j, i + 0.1, text_labels[i][j],
                 verticalalignment='top',
                 horizontalalignment="center",
                 fontsize=12,
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Print accuracy and precision
    print('Accuracy: ', accuracy_score(y_test, y_pred, normalize=True))
    print('Precision: ', precision_score(y_test, y_pred, average='macro'))
    print('Roc-Auc: ', roc_auc_score(y_test, y_pred))
    # Plot non-normalized confusion matrix


def run_test(X_, Y_, model_path, image_size):
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
    print('Restoring Model and Building Confusion Matrix')
    class_names = ['Not Hotdog', 'Hotdog']
    y_test, y_pred, cm = model_predictions(X_, Y_, model_path, image_size)
    print_cm(cm, class_names)

