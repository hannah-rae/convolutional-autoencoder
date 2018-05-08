import cv2
import numpy as np
from glob import glob
from random import shuffle

# def load_test_gray2rgb():
#     test_dir = '/home/hannah/data/mcam_vae/cdf_test_gray'
#     num_images = len(glob(test_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144, 160, 3))
#     for i, img_fn in enumerate(glob(test_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         dataset[i] = image
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def load_test_earth():
#     test_dir = '/home/hannah/data/mcam_vae/earth'
#     num_images = len(glob(test_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144, 160, 3))
#     for i, img_fn in enumerate(glob(test_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         image = image[:144, :160, :]
#         dataset[i] = image
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def load_test_pups():
#     test_dir = '/home/hannah/data/mcam_vae/puppies'
#     num_images = len(glob(test_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144, 160, 3))
#     for i, img_fn in enumerate(glob(test_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         image = image[7:151, 7:167, :]
#         dataset[i] = image
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def load_mcam_gray():
#     train_dir = '/home/hannah/data/mcam_vae/train'
#     num_images = len(glob(train_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144*144))
#     for i, img_fn in enumerate(glob(train_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
#         image = image[:, 7:151]
#         dataset[i] = image.flatten()
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def test_mcam_slices(i_len=36, j_len=40):
#     n = 16
#     def get_slices(image):
#         slices = np.zeros((n, i_len, j_len, 3))
#         idx = 0
#         for i in range(4):
#             for j in range(4):
#                 sl = image[i*i_len:i*i_len+i_len, j*j_len:j*j_len+j_len]
#                 slices[idx] = sl
#                 idx = idx + 1
#         return slices

#     test_dir = '/home/hannah/data/mcam_vae/cdf_test'
#     num_images = len(glob(test_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images*n, i_len, j_len, 3))
#     for i, img_fn in enumerate(glob(test_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         image_slices = get_slices(image)
#         dataset[i*n:i*n+n] = image_slices
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def load_mcam_slices(i_len=36, j_len=40):
#     n = 16
#     def get_slices(image):
#         slices = np.zeros((n, i_len, j_len, 3))
#         idx = 0
#         for i in range(4):
#             for j in range(4):
#                 sl = image[i*i_len:i*i_len+i_len, j*j_len:j*j_len+j_len]
#                 slices[idx] = sl
#                 idx = idx + 1
#         return slices

#     train_dir = '/home/hannah/data/mcam_vae/train'
#     num_images = len(glob(train_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images*n, i_len, j_len, 3))
#     for i, img_fn in enumerate(glob(train_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         image_slices = get_slices(image)
#         dataset[i*n:i*n+n] = image_slices
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def load_test_gray():
#     test_dir = '/home/hannah/data/mcam_vae/cdf_test'
#     num_images = len(glob(test_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144*144))
#     for i, img_fn in enumerate(glob(test_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)
#         image = image[:, 7:151]
#         dataset[i] = image.flatten()
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# def load_mcam_rgb():
#     train_dir = '/home/hannah/data/mcam_vae/train'
#     num_images = len(glob(train_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144, 160, 3))
#     for i, img_fn in enumerate(glob(train_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         # im_red = image[:,:,2]
#         # im_blue = image[:,:,0]
#         # image[:,:,0] = im_red
#         # image[:,:,2] = im_blue
#         # dataset[i,:,:,:] = image
#         dataset[i] = image
#         #dataset[i] = image.flatten()
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset

# TODO: Make train dir a flag.
# WARNING: Make sure when training on UDRs we are including the holdout_hardware set.
train_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/train'
hw_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/holdout_hardware'
train_examples = glob(train_dir + '/*')
train_examples = train_examples + glob(hw_dir + '/*')
# train_dir = '/home/hannah/data/mcamrdrs/train/cropped'
# train_examples = glob(train_dir + '/*.npy')
np.random.shuffle(train_examples)
num_train_ex = len(train_examples)
# TODO: Make epoch size a flag. Currently this is hardcoded for convenience.
train_examples = train_examples*30

def next_batch_6f(batchsize=5):
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold the batch of numpy arrays
    batch = np.zeros((batchsize, 64, 64, 6))
    # Store the images in the batch array
    for i in range(batchsize):
        ex = train_examples.pop()
        batch[i] = np.load(ex)

    #print 'this batch ', batch.shape
    #print 'train examples remaining ', len(train_examples)
    return batch

def load_mcam_6f_train(product):
    if product == 'UDR':
        train_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/train'
        hw_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/holdout_hardware'
        train_names = glob(train_dir + '/*')
        train_names = train_names + glob(hw_dir + '/*')
    elif product == 'RDR':
        train_dir = '/home/hannah/data/mcamrdrs/train/cropped'
        train_names = glob(train_dir + '/*')
    # np.random.shuffle(train_names)
    # train_names = train_names[:100]
    num_images = len(train_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(train_names):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset, [name.split('/')[-1][:-4] for name in train_names]

def load_mcam_6f_test():
    test_dir = '/home/hannah/data/mcam_Lall_Rall_64x64/test'
    test_names = glob(test_dir + '/*')
    num_images = len(test_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(test_names):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset#, [name.split('/')[-1][:-4] for name in test_names]

def load_mcam_DW_test_subject(subject):
    test_dir = '/home/hannah/data/mcam_multispec_DW_subject/' + subject
    test_names = glob(test_dir + '/*.npy')
    num_images = len(test_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(test_names):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset, [name.split('/')[-1][:-4] for name in test_names]

def load_mcam_DW_test_MR():
    test_dir = '/home/hannah/data/mcam_multispec_DW'
    test_names = glob(test_dir + '/*R*.npy')
    num_images = len(test_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(test_names):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset, [name.split('/')[-1][:-4] for name in test_names]

def load_mcam_DW_test_ML():
    test_dir = '/home/hannah/data/mcam_multispec_DW'
    test_names = glob(test_dir + '/*L*.npy')
    num_images = len(test_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(test_names):
        dataset[i] = np.load(img_fn)
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset, [name.split('/')[-1][:-4] for name in test_names]

def load_mcam_DW_test(product):
    if product == 'UDR':
        test_dir = '/home/hannah/data/mcam_multispec_DW'
    elif product == 'RDR':
        test_dir = '/home/hannah/data/mcamrdrs/expert/cropped'
    test_names = glob(test_dir + '/*.npy')
    num_images = len(test_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(test_names):
        im = np.load(img_fn)
        if im.shape != (64,64,6):
            continue
        else:
            dataset[i] = im
    print dataset.shape
    #np.random.shuffle(dataset)
    return dataset, [name.split('/')[-1][:-4] for name in test_names]

def load_explanation_images(path_to_results):
    image_dirs = glob(path_to_results+'/*')
    images = [glob(d+'/*explanation*')[0] for d in image_dirs] # N x 64 x 64 array
    num_images = len(images)
    # Each image is stored a 64 x 64 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 1), dtype=np.float32)
    # Store the images in the dataset array
    for i, img_fn in enumerate(images):
        dataset[i,:,:,0] = np.load(img_fn).astype(np.float32)
    print dataset.shape
    np.random.shuffle(dataset)
    return dataset

def load_diff_images(path_to_results):
    image_dirs = glob(path_to_results+'/*')
    images = [glob(d+'/*diff_6f*')[0] for d in image_dirs] # N x 64 x 64 x 6 array
    num_images = len(images)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6), dtype=np.float32)
    # Store the images in the dataset array
    for i, img_diff in enumerate(images):
        dataset[i] = np.load(img_diff).astype(np.float32)
    print dataset.shape
    np.random.seed(42) # Set the seed so we always get the same shuffle order
    np.random.shuffle(dataset)
    return dataset

def load_test_images():
    test_dir = '/home/hannah/data/mcam_Lall_Rall_udrs_sol1667to1925/cropped'
    test_names = glob(test_dir + '/*.npy')
    num_images = len(test_names)
    # Each image is stored a 64 x 64 x 6 numpy array
    # Create an empy array to hold all the numpy arrays
    dataset = np.zeros((num_images, 64, 64, 6))
    # Store the images in the dataset array
    for i, img_fn in enumerate(test_names):
        im = np.load(img_fn)
        if im.shape != (64,64,6):
            continue
        else:
            dataset[i] = im
    print dataset.shape
    # np.random.seed(42) # Set the seed so we always get the same shuffle order
    # np.random.shuffle(dataset)
    return dataset, [name.split('/')[-1][:-4] for name in test_names]

# def load_test_rgb():
#     test_dir = '/home/hannah/data/mcam_vae/cdf_test'
#     num_images = len(glob(test_dir + '/*'))
#     # Create an empty array to hold all the downsampled images
#     dataset = np.zeros((num_images, 144, 160, 3))
#     for i, img_fn in enumerate(glob(test_dir + '/*')):
#         image = cv2.imread(img_fn, cv2.IMREAD_COLOR)
#         # im_red = image[:,:,2]
#         # im_blue = image[:,:,0]
#         # image[:,:,0] = im_red
#         # image[:,:,2] = im_blue
#         # dataset[i,:,:,:] = image
#         dataset[i] = image
#         #dataset[i] = image.flatten()
#     print dataset.shape
#     np.random.shuffle(dataset)
#     return dataset