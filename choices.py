import time

import pandas as pd

import check
import convolutional_nn_valid as cnn
import cnn_working_3 as cnn4
import cnn_working as cnn2
import deeper_cnn_valid as cnn5
import cnn_working_2 as cnn3
import get_raw_images as scrape_images
import image_augmentation as ia
import remove_null_images as rm
import restore_to_test as restore_test

start_time = time.time()
paths = ['not_hotdog',
         'hotdog']

model_path = "conv_model/model.ckpt"


def choices():
    while True:
        print(
            'Type "run_all" to run all scripts \nType "train" to just train model \nType "test" to test saved model \n'
            'Or "break" to break\n')
        user_input = input('What would you like to run: ')
        # user_input = "'" + user_input + "'"

        if user_input == 'run_all':
            warning_input = input('WARNING: Are you sure you would like to run all?\n'
                                  'This can take up to 6 hours scrapping web for images.\n'
                                  'Do you want to run all? (y/n)')
            if warning_input == 'y':
                do_run_all()
            else:
                break

        if user_input == 'train':
            do_run_train()

        if user_input == 'test':
            do_run_test()

        if user_input == 'break':
            break

        if user_input is not ('run_all', 'train', 'test', 'break'):
            print('Please enter either "run_all", "train", or "test" \n')
            print('Try again \n')
            continue

        else:
            break


def do_run_all():
    image_size = 28
    while True:
        epoch_param = input('How many epochs would you like to run? \n')
        if epoch_param == 'break':
            break
        try:
            epoch_param = int(epoch_param)
        except ValueError:
            print('Please enter valid integer')
            continue
        else:
            break
    while True:
        learn_param = input('What do you want your learning rate would you like to use? \n')
        if learn_param == 'break':
            break
        try:
            learn_param = float(learn_param)
        except ValueError:
            print('Please enter valid float')
        else:
            break
    # This function will scrape images from Image.net
    scrape_images.read_images_to_folder()

    # # Removing all of the invalid or null images
    rm.remove_invalid(paths)

    # Rotating and or blurring images to create larger and balanced classes
    x, y = ia.load_data(image_size, 15000)

    # Run convolutional neural network
    total_loss, accuracy = cnn5.run(x, y, epochs=epoch_param, learning_rate=learn_param)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path, image_size)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path, image_size)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))


def do_run_train():
    image_size = 28
    while True:
        epoch_param = input('How many epochs would you like to run? \n')
        if epoch_param == 'break':
            break
        try:
            epoch_param = int(epoch_param)
        except ValueError:
            print('Please enter valid integer')
            continue
        else:
            break
    while True:
        learn_param = input('What do you want your learning rate would you like to use? \n')
        if learn_param == 'break':
            break
        try:
            learn_param = float(learn_param)
        except ValueError:
            print('Please enter valid float')
        else:
            break

    # Rotating and or blurring images to create larger and balanced classes
    if image_size <= 56:
        try:
            x = pd.read_csv('X.csv', sep=',')
            y = pd.read_csv('y.csv', sep=',')
        except:
            print('x or y csv does not exist')
            x, y = ia.load_data(image_size, 15000)
    else:
        x, y = ia.load_data(image_size, 15000)

    # Run convolutional neural network
    total_loss, accuracy = cnn5.run(x, y, epochs=epoch_param, learning_rate=learn_param, image_size=image_size)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path, image_size)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path, image_size)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))


def do_run_test():
    image_size = 28
    # Rotating and or blurring images to create larger and balanced classes
    x, y = ia.load_data(image_size, 15000)

    # Restore convolutional neural network and build confusion matrix
    restore_test.run_test(x, y, model_path, image_size)

    # Test the neural network on the hotdog and not_hotdog jpg
    check.run(model_path, image_size)

    end_time = time.time()
    print('Total Run Time {}'.format(round(end_time - start_time, 0)))
