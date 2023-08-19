from scipy import misc
from PIL import Image
from skimage import exposure
from sklearn import svm
import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
import pywt
import tensorflow as tf


def _filter_kernel_mf_fdog(L, sigma, t = 3, mf = True):
    dim_y = int(L)
    dim_x = 2 * int(t * sigma)
    arr = np.zeros((dim_y, dim_x), 'f')
    ctr_x = dim_x / 2 
    ctr_y = int(dim_y / 2.)
    it = np.nditer(arr, flags=['multi_index'])
    while not it.finished:
        arr[it.multi_index] = it.multi_index[1] - ctr_x
        it.iternext()
    two_sigma_sq = 2 * sigma * sigma
    sqrt_w_pi_sigma = 1. / (sqrt(2 * pi) * sigma)
    if not mf:
        sqrt_w_pi_sigma = sqrt_w_pi_sigma / sigma ** 2

    def k_fun(x):
        return sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    def k_fun_derivative(x):
        return -x * sqrt_w_pi_sigma * exp(-x * x / two_sigma_sq)

    if mf:
        kernel = k_fun(arr)
        kernel = kernel - kernel.mean()
    else:
        kernel = k_fun_derivative(arr)

    return cv2.flip(kernel, -1) 


def show_images(images,titles=None, scale=1.3):
    """Display a list of images"""
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)  # Make subplot
        if image.ndim == 2:  # Is image grayscale?
            plt.imshow(image, cmap = cm.Greys_r)
        else:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        a.set_title(title)
        plt.axis("off")
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches(), dtype=np.float) * n_ims / scale)
    plt.show()


def gaussian_matched_filter_kernel(L, sigma, t = 3):
    '''K =  1/(sqrt(2 * pi) * sigma ) * exp(-x^2/2sigma^2), |y| <= L/2, |x| < s * t'''
    return _filter_kernel_mf_fdog(L, sigma, t, True)


def create_matched_filter_bank(K, n = 12):
    """Creating a matched filter bank using the kernel generated from the above functions"""
    rotate = 180 / n
    center = (K.shape[1] / 2, K.shape[0] / 2)
    cur_rot = 0
    kernels = [K]
    for i in range(1, n):
        cur_rot += rotate
        r_mat = cv2.getRotationMatrix2D(center, cur_rot, 1)
        k = cv2.warpAffine(K, r_mat, (K.shape[1], K.shape[0]))
        kernels.append(k)
    return kernels


def apply_filters(im, kernels):
    """Given a filter bank, apply them and record maximum response"""
    images = np.array([cv2.filter2D(im, -1, k) for k in kernels])
    return np.max(images, 0)


def create_new_matched_filter_bank():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kern = cv2.getGaborKernel((ksize, ksize), 6, theta,12, 0.37, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters


def preprocess_image(img):
    """Function to preprocess the image"""
    immatrix = []
    new_row = 2848
    new_col = 4288
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = cv2.equalizeHist(img_gray) 
    processed_image = tf.image.resize(np.reshape(equ, (2848, 4288, 1)), (new_row,new_col)).numpy()
    immatrix.append(np.array(processed_image).flatten())
    imm_dwt = []
    for equ in immatrix:
        equ = equ.reshape((new_row,new_col))
        coeffs = pywt.dwt2(equ, 'haar')
        equ2 = pywt.idwt2(coeffs, 'haar')
        imm_dwt.append(np.array(equ2).flatten())

    gf = gaussian_matched_filter_kernel(20, 5)
    bank_gf = create_matched_filter_bank(gf, 4)
    imm_gauss = []
    for equ2 in imm_dwt:
        equ2 = equ2.reshape((new_row,new_col))
        equ3 = apply_filters(equ2, bank_gf)
        imm_gauss.append(np.array(equ3).flatten())

    bank_gf = create_new_matched_filter_bank()
    imm_gauss2 = []
    for equ2 in imm_dwt:
        equ2 = equ2.reshape((new_row,new_col))
        equ3 = apply_filters(equ2, bank_gf)
        imm_gauss2.append(np.array(equ3).flatten())

    e_ = equ3
    np.shape(e_)
    e_=e_.reshape((-1,4))
    np.shape(e_)
    img = equ3
    Z = img.reshape((-1,4))
    Z = np.float32(Z)  # convert to np.float32
    k = cv2.KMEANS_PP_CENTERS

    # Define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, k)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    imm_kmean = []
    for equ3 in imm_gauss2:
        img = equ3.reshape((new_row,new_col))
        Z = img.reshape((-1,4))
        Z = np.float32(Z)
        k = cv2.KMEANS_PP_CENTERS

        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, k)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        imm_kmean.append(np.array(res2).flatten())

    index = 0
    processed_image = tf.image.resize(np.reshape(imm_kmean[index], (new_row,new_col,1)), (4,4)).numpy()
    processed_image = processed_image/255
    processed_image = processed_image.flatten()
    processed_image.resize(4,4)
    return processed_image
