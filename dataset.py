import cv2
import numpy as np
from glob import glob
from random import shuffle

def load_test_gray2rgb():
    test_dir = '/home/hannah/data/mcam_vae/cdf_test_gray'
    num_images = len(glob(test_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144, 160, 3))
    for i, img_fn in enumerate(glob(test_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        dataset[i] = image
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_test_earth():
    test_dir = '/home/hannah/data/mcam_vae/earth'
    num_images = len(glob(test_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144, 160, 3))
    for i, img_fn in enumerate(glob(test_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        image = image[:144, :160, :]
        dataset[i] = image
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_test_pups():
    test_dir = '/home/hannah/data/mcam_vae/puppies'
    num_images = len(glob(test_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144, 160, 3))
    for i, img_fn in enumerate(glob(test_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        image = image[7:151, 7:167, :]
        dataset[i] = image
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_mcam_gray():
    train_dir = '/home/hannah/data/mcam_vae/train'
    num_images = len(glob(train_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144*144))
    for i, img_fn in enumerate(glob(train_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        image = image[:, 7:151]
        dataset[i] = image.flatten()
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def test_mcam_slices(i_len=36, j_len=40):
    n = 16
    def get_slices(image):
        slices = np.zeros((n, i_len, j_len, 3))
        idx = 0
        for i in range(4):
            for j in range(4):
                sl = image[i*i_len:i*i_len+i_len, j*j_len:j*j_len+j_len]
                slices[idx] = sl
                idx = idx + 1
        return slices

    test_dir = '/home/hannah/data/mcam_vae/cdf_test'
    num_images = len(glob(test_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images*n, i_len, j_len, 3))
    for i, img_fn in enumerate(glob(test_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        image_slices = get_slices(image)
        dataset[i*n:i*n+n] = image_slices
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_mcam_slices(i_len=36, j_len=40):
    n = 16
    def get_slices(image):
        slices = np.zeros((n, i_len, j_len, 3))
        idx = 0
        for i in range(4):
            for j in range(4):
                sl = image[i*i_len:i*i_len+i_len, j*j_len:j*j_len+j_len]
                slices[idx] = sl
                idx = idx + 1
        return slices

    train_dir = '/home/hannah/data/mcam_vae/train'
    num_images = len(glob(train_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images*n, i_len, j_len, 3))
    for i, img_fn in enumerate(glob(train_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        image_slices = get_slices(image)
        dataset[i*n:i*n+n] = image_slices
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_test_gray():
    test_dir = '/home/hannah/data/mcam_vae/cdf_test'
    num_images = len(glob(test_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144*144))
    for i, img_fn in enumerate(glob(test_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
        image = image[:, 7:151]
        dataset[i] = image.flatten()
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_mcam_rgb():
    train_dir = '/home/hannah/data/mcam_vae/train'
    num_images = len(glob(train_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144, 160, 3))
    for i, img_fn in enumerate(glob(train_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        # im_red = image[:,:,2]
        # im_blue = image[:,:,0]
        # image[:,:,0] = im_red
        # image[:,:,2] = im_blue
        # dataset[i,:,:,:] = image
        dataset[i] = image
        #dataset[i] = image.flatten()
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_mcam_6f():
    train_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/train'
    num_images = len(glob(train_dir + '/*')[:20000])
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(glob(train_dir + '/*')[:20000]):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_mcam_6f_test():
    test_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/test'
    num_images = len(glob(test_dir + '/*')[:100])
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(glob(test_dir + '/*')[:100]):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset

def load_test_rgb():
    test_dir = '/home/hannah/data/mcam_vae/cdf_test'
    num_images = len(glob(test_dir + '/*'))
    # Create an empty array to hold all the downsampled images
    dataset = np.zeros((num_images, 144, 160, 3))
    for i, img_fn in enumerate(glob(test_dir + '/*')):
        image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
        # im_red = image[:,:,2]
        # im_blue = image[:,:,0]
        # image[:,:,0] = im_red
        # image[:,:,2] = im_blue
        # dataset[i,:,:,:] = image
        dataset[i] = image
        #dataset[i] = image.flatten()
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset