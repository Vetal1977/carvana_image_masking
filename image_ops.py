import numpy as np
from skimage import io
import cv2

IMAGE_WIDTH_TRAIN = 384
IMAGE_HEIGHT_TRAIN = 256
IMAGE_WIDTH_ORIG = 1918
IMAGE_HEIGHT_ORIG = 1280


def rle(mask_img):
    """
    Creates Run Length Encoding string for the mask image
    :param mask_img: mask image is a numpy array, where 1 means 'mask', 0 - 'background'
    :return: RLE as string
    """
    INIT_RLE_PIXEL_POS = -2
    INIT_NO_OF_PIXELS = 0

    mask_pixel_positions = np.where(mask_img.flatten() == 1)[0]
    rle_list = []
    prev_mask_pixel_pos = INIT_RLE_PIXEL_POS

    for cur_mask_pixel_pos in mask_pixel_positions:
        if cur_mask_pixel_pos > prev_mask_pixel_pos + 1:
            rle_list.extend((cur_mask_pixel_pos + 1, INIT_NO_OF_PIXELS))
        rle_list[-1] += 1
        prev_mask_pixel_pos = cur_mask_pixel_pos

    return ' '.join([str(i) for i in rle_list])


def load_and_resize_image(img_file_path, height, width):
    """
    Loads and resizes image (scales up or down)
    :param img_file_path: absolute or relative path to the image (JPEG or GIF)
    :param height: target height of the image
    :param width: target width of the image
    :return: image as numpy array in format (height, width, channels)
    """
    img = io.imread(img_file_path)

    img = resize_image(img, height, width)

    if img.ndim == 2: # mask
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))

    img = img.astype(np.float16)
    img /= 255.0

    return img


def resize_image(img, height, width):
    """
    Resizes the image (scales up or down)
    :param img: source image as numpy array in format (source_height, source_width, channels)
    :param height: target height of the image
    :param width: target width of the image
    :return: image as numpy array in format (height, width, channels)
    """
    return cv2.resize(img, (width, height), cv2.INTER_LINEAR)
