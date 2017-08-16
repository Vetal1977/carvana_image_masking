import os
import datetime
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from skimage import io
import numpy as np
import carvana_model as cm
import data_preprocessing as dpp
import image_ops
from pandas import DataFrame

USE_BATCH_GENERATOR = True
MODELS_DIR = './models/'
TEST_DIR = './test/'
SMALL_TEST_DIR = './test_small/'
SUBMISSIONS_DIR = './submissions/'


def create_output_dirs():
    """
    Creates output directories for model and submission
    """
    if not os.path.isdir(MODELS_DIR):
        os.mkdir(MODELS_DIR)
    if not os.path.isdir(SUBMISSIONS_DIR):
        os.mkdir(SUBMISSIONS_DIR)


def train(model, model_path, train_files, val_files, train_masks, val_masks, epochs, batch_size):
    """
    Trains Carvana model with keras; either using a batch generator or
    just a usual fit function
    :param model: Carvana U-Net model
    :param model_path: path of the model file (stored in HDF format)
    :param train_files: list of train image files
    :param val_files: list of validation image files
    :param train_masks: list of train mask files
    :param val_masks: list of validation mask files
    :param epochs: Number of epochs
    :param batch_size: train batch size; should be small
    (ca. 8 for a GPU with 6 GB RAM) 
    because of the image size
    """
    model.compile(
        optimizer=Adam(lr=0.0008),
        loss='binary_crossentropy',
        metrics=['accuracy', cm.dice_coeff])

    if USE_BATCH_GENERATOR:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=2),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, verbose=2),
        ]

        model.fit_generator(
            generator=dpp.batch_generator(train_files, batch_size),
            steps_per_epoch=len(train_files)/batch_size,
            validation_data=dpp.batch_generator(val_files, batch_size),
            validation_steps=len(val_files)/batch_size,
            epochs=epochs,
            callbacks=callbacks)
    else:
        train_images, train_mask_images = dpp.load_images(train_files, train_masks, 32)
        val_images, val_mask_images = dpp.load_images(val_files, val_masks, 8)
        print("Number of train images: {}".format(len(train_images)))
        print("Number of mask images: {}".format(len(train_mask_images)))

        model.fit(
            x=train_images,
            y=train_mask_images,
            epochs=epochs,
            validation_data=(val_images, val_mask_images),
            batch_size=batch_size)


def create_upscaled_mask(pred_mask):
    """
    Creates upscaled predicted mask
    :param pred_mask: predicted mask as a numpy array in format
    (train height, train width, 1)
    :return: upscaled predicted mask as a numpy array in format
    (original height, original width, 1) where 1 means 'mask', 0 - 'background'
    """
    upscaled_pred = pred_mask
    upscaled_pred[upscaled_pred > 0.5] = 1
    upscaled_pred[upscaled_pred <= 0.5] = 0
    upscaled_pred = image_ops.resize_image(
        (255*upscaled_pred).astype(np.uint8),
        image_ops.IMAGE_HEIGHT_ORIG,
        image_ops.IMAGE_WIDTH_ORIG)
    upscaled_pred = upscaled_pred/255.0
    upscaled_pred[upscaled_pred > 0.5] = 1
    upscaled_pred[upscaled_pred <= 0.5] = 0

    return upscaled_pred


def predict_and_make_submission_file(model, no_of_pred_at_once):
    """
    Makes prediction for test images and creates a submission CSV file
    :param model: trained Carvana model
    """
    df = DataFrame(columns=('img', 'rle_mask'))

    # list test image files and run prediction on all of them
    test_files = os.listdir(TEST_DIR)
    no_of_test_files = len(test_files)
    start = 0
    while start < no_of_test_files:
        end = min(start + no_of_pred_at_once, no_of_test_files)
        predict_files = test_files[start:end]
        
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{}: Processing {} - {} of {} - {}...'.format(
            now, start, end, no_of_test_files, predict_files[0]))

        test_images = []
        pred_names = []
        for _, test_file in enumerate(predict_files):
            # resize image for prediction - it must have the same size
            # as by training
            test_img = image_ops.load_and_resize_image(
                TEST_DIR + test_file,
                image_ops.IMAGE_HEIGHT_TRAIN,
                image_ops.IMAGE_WIDTH_TRAIN)
            test_images.append(test_img)
            pred_names.append(test_file)

        # predict
        test_images = np.array(test_images)
        predictions = model.predict(test_images)

        for ii in range(predictions.shape[0]):
            # upscale predicted mask to form RLE string
            upscaled_pred = create_upscaled_mask(predictions[ii])

            # create RLE string...
            rle_string = image_ops.rle(upscaled_pred)

            # ... and save it into data frame
            df.loc[-1] = (pred_names[ii], rle_string)
            df.index += 1

        start += no_of_pred_at_once

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    RES_FILE_NAME = SUBMISSIONS_DIR + '{}_submission.csv.gz'.format(now)
    df.to_csv(RES_FILE_NAME, index=False, compression='gzip')
    print('Results are written into ', RES_FILE_NAME)


def predict_and_save_images(model, no_of_pred_at_once):
    """
    [For debug purpose] Makes prediction for test images and saves masks
    and images to the disk. Must be used only for a small number of test images!
    :param model: trained Carvana model
    """
    # list files in the train folder with a small number of images
    test_files = os.listdir(SMALL_TEST_DIR)
    no_of_test_files = len(test_files)
    start = 0
    while start < no_of_test_files:
        end = min(start + no_of_pred_at_once, no_of_test_files)
        predict_files = test_files[start:end]
        
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('{}: Processing {} - {} of {} - {}...'.format(
            now, start, end, no_of_test_files, predict_files[0]))

        test_images = []
        for ii, test_file in enumerate(predict_files):
            # resize image for prediction - it must have the same size
            # as by training
            test_img = image_ops.load_and_resize_image(
                TEST_DIR + test_file,
                image_ops.IMAGE_HEIGHT_TRAIN,
                image_ops.IMAGE_WIDTH_TRAIN)
            test_images.append(test_img)

            # calculate upscaled source image (to demo, how the upscaling is working)
            upscaled_src = image_ops.resize_image(
                (255*test_img).astype(np.uint8),
                image_ops.IMAGE_HEIGHT_ORIG,
                image_ops.IMAGE_WIDTH_ORIG)
            io.imsave('./predictions/{:03d}_{}.jpg'.format(start + ii, test_file), upscaled_src)

        # predict
        test_images = np.array(test_images)
        predictions = model.predict(test_images)

        for ii in range(predictions.shape[0]):
            # calcualted upscaled predicted mask
            upscaled_pred = predictions[ii]
            upscaled_pred[upscaled_pred > 0.5] = 1
            upscaled_pred[upscaled_pred <= 0.5] = 0
            upscaled_pred = image_ops.resize_image(
                (255*upscaled_pred).astype(np.uint8),
                image_ops.IMAGE_HEIGHT_ORIG,
                image_ops.IMAGE_WIDTH_ORIG)
            io.imsave('./predictions/{:03d}_pred.jpg'.format(start + ii), np.squeeze(upscaled_pred))

        start += no_of_pred_at_once


if __name__ == '__main__':
    create_output_dirs()

    carvana_model = cm.UNET_Carvana(dropout_val=0.1, batch_norm=True)

    # train or load the model
    model_path = MODELS_DIR + 'cravana_unet.h5'
    if os.path.isfile(model_path):
        print('Loading model from last point: ', model_path)
        carvana_model.load_weights(model_path)
    else:
        print('Training the model...')
        train_files, val_files, train_masks, val_masks = dpp.list_train_and_validation_image_files()
        train(
            model=carvana_model,
            model_path=model_path,
            train_files=train_files,
            val_files=val_files,
            train_masks=train_masks,
            val_masks=val_masks,
            epochs=15,
            batch_size=8)

    # predict_and_save_images(carvana_model, 20)
    predict_and_make_submission_file(carvana_model, 20)
