import numpy as np
import matplotlib.pyplot as plt
import scipy

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc # pip install Pillow
import matplotlib.pylab as pylab
from tqdm import tqdm

import numpy as np

import cv2

from skimage.filters.rank import entropy
from skimage.morphology import square

import math



class DCT_helper:
    def __init__(self):
        pass

    def dct2(self, a):
        return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')    # fft or fftpack jsut check once on which is which

    def idct2(self ,a):
        return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


    def local_DCT(self,im, block_size=8):
        imsize = im.shape  # h*w
        dct = np.zeros(imsize)
        # https://stackoverflow.com/questions/30597869/what-does-np-r-do-numpy
        # Do 8x8 DCT on image (in-place) using 8*8 blocks
        for i in np.r_[0:imsize[0]:block_size]:
            for j in np.r_[0:imsize[1]:block_size]:
                dct[i:(i + block_size), j:(j + block_size)] = self.dct2(im[i:(i + block_size), j:(j + block_size)])
        return dct










    def calculate_standard_deviation(self,im, block_size=8, mask_thresh=0.1, use_mask_thresh=True):

        imsize = im.shape
        no_of_i = (np.r_[0:imsize[0]:block_size].shape)[0] + 1
        no_of_j = (np.r_[0:imsize[1]:block_size].shape)[0] + 1
        std_deviation_matrix = np.zeros([no_of_i, no_of_j])
        # mask    = np.zeros([no_of_i,no_of_j])
        mask = np.zeros(imsize)

        white_block = np.ones([block_size, block_size])
        cnti = 0
        cntj = 0

        # Do 8x8 DCT on image (in-place) using 8*8 blocks
        for i in np.r_[0:imsize[0]:block_size]:
            cntj = 0
            for j in np.r_[0:imsize[1]:block_size]:
                std_deviation_matrix[cnti][cntj] = np.std(im[i:(i + block_size), j:(j + block_size)])  # applying standard deviation function to a image block
                cntj = cntj + 1
            cnti = cnti + 1

        mask_thresh = np.amax(std_deviation_matrix)/10
        cnti = 0
        for i in np.r_[0:imsize[0]:block_size]:
            cntj = 0
            for j in np.r_[0:imsize[1]:block_size]:
                if std_deviation_matrix[cnti][cntj] < mask_thresh:  # if std greater than thresh not a homogenous region
                    mask[i:(i + block_size), j:(j + block_size)] = np.ones(im[i:(i + block_size), j:(j + block_size)].shape)
                cntj = cntj + 1
            cnti = cnti + 1

        f = plt.figure()
        plt.title("Standard Deviation Matrix")
        plt.imshow(std_deviation_matrix)

        f = plt.figure()
        plt.title("Mask with homogenous highlighted")
        plt.imshow(mask)

        return std_deviation_matrix,mask