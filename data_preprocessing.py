import os
from sklearn.model_selection import train_test_split
import image_ops
import numpy as np

# Number of image channels (for example 3 in case of RGB, or 1 for grayscale images)
INPUT_CHANNELS = 3
# Number of output masks (1 in case you predict only one type of objects)
OUTPUT_MASK_CHANNELS = 1

TRAIN_DIR = './train/'
MASK_DIR = './train_masks/'

np.random.seed(1234)


def list_train_and_validation_image_files():
    """
    Lists image files from the train and mask directories and splits them into
    train and validation sets
    :return: quadruple of lists
    (train image files, validation image files, train masks, validation masks)
    """
    train_files = os.listdir(TRAIN_DIR)
    mask_files = []
    for _, train_file in enumerate(train_files):
        mask_file = train_file.split('.')[0] + '_mask.gif'
        mask_files.append(mask_file)

    train_files, val_files, train_masks, val_masks = train_test_split(
        train_files,
        mask_files,
        test_size=0.3)

    return train_files, val_files, train_masks, val_masks


def load_images(image_files, mask_files, max_number_of_images_to_load=-1):
    """
    Loads images and masks from files
    :param images_files: list of image files
    :param mask_files: list of mask files
    :param max_number_of_images_to_load: max. number of images and masks
    to load. Value <= 0 means 'load all'
    :return: tuple of numpy arrays (images, masks). Each image and mask has a format
    of (height, width, channels)
    """
    images = []
    masks = []

    no_of_images = max_number_of_images_to_load \
                    if max_number_of_images_to_load > 0 \
                    else len(image_files)
    images = np.zeros(
        (no_of_images, image_ops.IMAGE_HEIGHT_TRAIN, image_ops.IMAGE_WIDTH_TRAIN, 3),
        dtype=np.float16)
    masks = np.zeros(
        (no_of_images, image_ops.IMAGE_HEIGHT_TRAIN, image_ops.IMAGE_WIDTH_TRAIN, 1),
        dtype=np.float16)

    for ii in range(no_of_images):
        print(image_files[ii])
        print(mask_files[ii])

        images[ii] = image_ops.load_and_resize_image(
            TRAIN_DIR + image_files[ii],
            image_ops.IMAGE_HEIGHT_TRAIN,
            image_ops.IMAGE_WIDTH_TRAIN)
        masks[ii] = image_ops.load_and_resize_image(
            MASK_DIR + mask_files[ii],
            image_ops.IMAGE_HEIGHT_TRAIN,
            image_ops.IMAGE_WIDTH_TRAIN)

    return images, masks


def batch_generator(files, batch_size):
    """
    Batch generator for keras fit_generator
    :param files: list of image files (e.g. train or validation)
    :param batch_size: batch size for training
    """
    batch_normal = batch_size
    np.random.shuffle(files)
    number_of_batches = np.floor(len(files) / batch_normal)
    counter = 0

    while True:
        batch_files = files[batch_normal * counter:batch_normal * (counter + 1)]
        image_list = []
        mask_list = []
        for image_file in batch_files:
            train_image_file = TRAIN_DIR + image_file
            mask_file = MASK_DIR + image_file.split('.')[0] + '_mask.gif'

            image = image_ops.load_and_resize_image(
                train_image_file,
                image_ops.IMAGE_HEIGHT_TRAIN,
                image_ops.IMAGE_WIDTH_TRAIN)
            mask = image_ops.load_and_resize_image(
                mask_file,
                image_ops.IMAGE_HEIGHT_TRAIN,
                image_ops.IMAGE_WIDTH_TRAIN)

            image_list.append(image)
            mask_list.append(mask)

        counter += 1

        image_list = np.array(image_list)
        mask_list = np.array(mask_list)

        yield image_list, mask_list
        if counter >= number_of_batches:
            np.random.shuffle(files)
            counter = 0
