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



class HIFST_utilities:

    def __init__(self):
        pass
    # Section 2.1 page 5801
    def dct2(self, a):
        return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')    # fft or fftpack jsut check once on which is which

    def idct2(self ,a):
        return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Section 2.1 : division of frequency bands into high medium and low frequency
    def extract_frequency_bands(self ,dct, block_size):
        high_frequency_bands = []
        medium_frequency_bands = []
        low_frequency_bands = []
        all_high_freq = []
        M = block_size - 1
        imsize = dct.shape  # h*w

        for i in r_[0:imsize[0]:block_size]:
            for j in r_[0:imsize[1]:block_size]:
                temphigh = []
                templow = []
                tempmed = []
                for k in range(M + 1):
                    for l in range(M + 1):
                        if i + k < imsize[0] and j + l < imsize[1]:
                            if k + l >= M:
                                temphigh.append(dct[i + k][j + l])
                                all_high_freq.append(dct[i + k][j + l])
                            elif k + l > M // 2:
                                tempmed.append(dct[i + k][j + l])
                            else:
                                templow.append(dct[i + k][j + l])
                low_frequency_bands.append(templow.copy())
                medium_frequency_bands.append(tempmed.copy())
                high_frequency_bands.append(temphigh.copy())
        return high_frequency_bands, medium_frequency_bands, low_frequency_bands, all_high_freq

    # Section 2.1 :  Marking the n*n DCT block 3:DC coefficient   2:Low freq  1: Medium Frequency   0: High frequency
    def FreqBands(self, MatrixSize):
        NewMatrix = np.zeros([MatrixSize, MatrixSize])
        for k in range(MatrixSize):
            for l in range(MatrixSize):
                if k + l >= MatrixSize - 1:
                    NewMatrix[k][l] = 0
                elif k + l >= (MatrixSize - 1) // 2:
                    NewMatrix[k][l] = 1
                else:
                    NewMatrix[k][l] = 2

        NewMatrix[0][0] = 3
        return NewMatrix




    # Section 2.2. :Using a gaussian kernel to add noise to the image we add gaussian noise which basically makes use of normal distribution
    # Matlab documentation of noise
    # https://in.mathworks.com/help/images/ref/imnoise.html
    def add_noise(self ,noise_typ, image):
        # Gaussian distribution an normal distribution is the same it gives the probabiluity within a range
        if noise_typ == "gauss":  # J = imnoise(I,'gaussian',m,var_gauss) adds Gaussian white noise with mean m and variance var_gauss. Matlab
            row, col = image.shape  # take image shape
            mean = 0
            var = 1  # variance of normal distribution
            sigma = var ** 0.5  # sigma is square root of variance
            gauss = np.random.normal(mean, sigma, (row, col))  # normal distribution
            #   print(gauss)
            gauss = gauss.reshape(row, col)
            noisy = image + gauss
            return noisy
        elif noise_typ == "salt&pepper":
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            out = np.copy(image)
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                      for i in image.shape]
            out[coords] = 1

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
            out[coords] = 0
            return out
        elif noise_typ == "poisson":
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))
            noisy = np.random.poisson(image * vals) / float(vals)
            return noisy
        elif noise_typ == "speckle":
            row, col, ch = image.shape
            gauss = np.random.randn(row, col, ch)
            gauss = gauss.reshape(row, col, ch)
            noisy = image + image * gauss
            return noisy


    # Section 2.2 (After adding gaussian noise ):  Last step of preprocessing where we get the gradient
    # Calculate Gradient and Direction of Gradient in image
    # Sobel Opencv Documentation
    # OpenCV provides three types of gradient filters or High-pass filters, Sobel, Scharr and Laplacian.
    # run cv2.Sobel twice on this image, we specify the direction of the derivative you want to find.

    def gradient_magnitude_direction(self, img):
        # Find x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0)  # sobelx is the derivative in the x direction
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1)  # sobely is the derivative in the y direction

        # Find magnitude and angle
        magnitude = np.sqrt(sobelx ** 2.0 + sobely ** 2.0)
        angle = np.arctan2(sobely, sobelx) * (180 / np.pi)
        return magnitude, angle


    #Normalize value of matrix between 0 to 1
    def mat_2_gray(self , matA):
        matA = np.double(matA)
        out = np.zeros(matA.shape, np.double)
        normalized = cv2.normalize(matA, out, 1.0, 0.0, cv2.NORM_MINMAX)
        return normalized


    #Calculate patch wise local entropy of an image
    # Section 2.2  equation 9a calculation of omega
    def local_entropy(self ,image, mat_size=7):
        ent = entropy(image, square(mat_size))
        return ent




    def TransformedDomainRecursiveFilter_Horizontal(self, I, D, sigma,debug=False):

        # Feedback coefficient (Appendix of our paper)
        a = math.exp(-math.sqrt(2) / sigma)

        F = I
        V = np.power(a, D)

        h, w = I.shape
        print("I shape :{}".format(I.shape))
        print("V shape :{}".format(V.shape))
        # Left -> Right filter.
        for i in range(1, w):

            F[:, i] = F[:, i] + np.multiply(V[:, i], (F[:, i - 1] - F[:, i]))

        # Right -> Left filter.
        for i in range(w - 2, -1, -1):
            F[:, i] = F[:, i] + np.multiply(V[:, i + 1], (F[:, i + 1] - F[:, i]))

        if (debug==True):
            f = plt.figure()
            plt.title("Output_image of TransformedDomainRecursiveFilter_Horizontal")
            plt.imshow(F)

        return F




    def image_transpose(self,I):

        h, w = I.shape

        T = np.zeros([w, h])

        # for i in range(c):
        T[:, :] = np.transpose(I[:, :])
        return T


    # Recursive edge preserving filter
    def RF(self,img, joint_image,sigma_s=15, sigma_r=0.25, num_iterations=3):

        J = joint_image
        h, w = joint_image.shape
        # Distancec between neighbouring samples
        dIcdx = np.diff(J, axis=1)
        dIcdy = np.diff(J, axis=0)

        dIdx = np.zeros([h, w])
        dIdy = np.zeros([h, w])

        # Compute the l1-norm distance of neighbor pixels.
        # for i in range(c):
        dIdx[:, 1:] = dIdx[:, 1:] + abs(dIcdx[:, :])
        dIdy[1:, :] = dIdy[1:, :] + abs(dIcdy[:, :])

        # Compute the derivatives of the horizontal and vertical domain transforms.
        dHdx = (1 + sigma_s / sigma_r * dIdx)
        dVdy = (1 + sigma_s / sigma_r * dIdy)

        # The vertical pass is performed using a transposed image.
        dVdy = np.transpose(dVdy)

        # Perform the filtering.

        N = num_iterations
        F = img

        sigma_H = sigma_s

        for i in range(num_iterations):
            # Compute the sigma value for this iteration (Equation 14 of paper).
            sigma_H_i = sigma_H * math.sqrt(3) * 2 ** (N - (i + 1)) / math.sqrt(4 ** N - 1)

            F = self.TransformedDomainRecursiveFilter_Horizontal(F, dHdx, sigma_H_i)
            F = self.image_transpose(F)

            F = self.TransformedDomainRecursiveFilter_Horizontal(F, dVdy, sigma_H_i)
            F = self.image_transpose(F)

        return F





    def fspecial_gauss2D(self ,shape = (3, 3), sigma=0.5):
        """
        2D gaussian mask - should give the same result as MATLAB's
        fspecial('gaussian',[shape],[sigma])
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h






    def Hifst(self,GrayImg, Noise,sig_s,sig_r,num_iter, slide=1, odd_Flag=1,debug=False ):


        print("GrayImg size {}".format(GrayImg.shape))
        noisy = self.add_noise("gauss", GrayImg)

        # https://in.mathworks.com/help/images/ref/imgaussfilt.html   Matlab imgauss
        gauss_blur = cv2.GaussianBlur(noisy, ksize=(0, 0), sigmaX=0.5, borderType=cv2.BORDER_REPLICATE)

        gradient_mag, gradient_dir = self.gradient_magnitude_direction(gauss_blur)

        # In the paper they took a 7*7 patch so flag=1
        if odd_Flag == 1:
            M_1 = (2 ** 3) - 1
            M_2 = (2 ** 4) - 1
            M_3 = (2 ** 5) - 1
            M_4 = (2 ** 6) - 1
        elif odd_Flag == 0:
            M_1 = (2 ** 3)
            M_2 = (2 ** 4)
            M_3 = (2 ** 5)
            M_4 = (2 ** 6)

        SelectedNumLayers = M_1 + M_2 + M_3 + M_4 + 1
        OutIndex1 = self.FreqBands(M_1)
        OutIndex2 = self.FreqBands(M_2)
        OutIndex3 = self.FreqBands(M_3)
        OutIndex4 = self.FreqBands(M_4)
        # print("\n__1__\n")
        # print(OutIndex1)
        # print("\n__2__\n")
        # print(OutIndex2)
        # print("\n__3__\n")
        # print(OutIndex3)
        # print("\n__4__\n")
        # print(OutIndex4)
        if debug == True:
            f = plt.figure()
            plt.title("Gaussian blur")
            plt.imshow(gauss_blur)
            f = plt.figure()
            plt.title("Magnitude")
            plt.imshow(gradient_mag)
            f = plt.figure()
            plt.title("angle")
            plt.imshow(gradient_dir)
            print("Before padding  shape :{}".format(gradient_mag.shape))

        gradient_mag_pad = np.pad(gradient_mag, (M_4 // 2, M_4 // 2), 'constant', constant_values=(0, 0))
        padded_size = gradient_mag_pad.shape
        if debug == True:
            print("Before padding  shape :{}".format(gradient_mag_pad.shape))

        non_pad_0 = padded_size[0] - max([M_1, M_2, M_3, M_4]) // 2 - max([M_1, M_2, M_3, M_4]) // 2
        non_pad_1 = padded_size[1] - max([M_1, M_2, M_3, M_4]) // 2 - max([M_1, M_2, M_3, M_4]) // 2
        L = np.zeros([non_pad_0, non_pad_1, SelectedNumLayers])
        n = -1

        print("L_shape {}".format(L.shape))

        for i in tqdm(range(max([M_1, M_2, M_3, M_4]) // 2, padded_size[0] - max([M_1, M_2, M_3, M_4]) // 2,slide)):  # 31 to 530 gradient mag without the padding
            m = -1

            n = n + 1

            for j in range(max([M_1, M_2, M_3, M_4]) // 2, padded_size[1] - max([M_1, M_2, M_3, M_4]) // 2, slide):
                m = m + 1

                #         Block selection
                Patch1 = gradient_mag_pad[i - (M_1 // 2):i + (M_1 // 2) + 1, j - (M_1 // 2):j + (M_1 // 2) + 1]
                Patch2 = gradient_mag_pad[i - (M_2 // 2):i + (M_2 // 2) + 1, j - (M_2 // 2):j + (M_2 // 2) + 1]
                Patch3 = gradient_mag_pad[i - (M_3 // 2):i + (M_3 // 2) + 1, j - (M_3 // 2):j + (M_3 // 2) + 1]
                Patch4 = gradient_mag_pad[i - (M_4 // 2):i + (M_4 // 2) + 1, j - (M_4 // 2):j + (M_4 // 2) + 1]

                # Computing DCTs
                DCT_Coef1 = abs(self.dct2(Patch1))

                DCT_Coef2 = abs(self.dct2(Patch2))

                DCT_Coef3 = abs(self.dct2(Patch3))

                DCT_Coef4 = abs(self.dct2(Patch4))
                # print("\n&&&&&&&&&&&&")
                # print(Patch1.shape)
                # print(OutIndex1.shape)

                H1 = DCT_Coef1[OutIndex1 == 0]

                H2 = DCT_Coef2[OutIndex2 == 0]

                H3 = DCT_Coef3[OutIndex3 == 0]

                H4 = DCT_Coef4[OutIndex4 == 0]

                #        Sorting
                # print(H1.shape)
                # print(H2.shape)
                # print(H3.shape)
                # print(H4.shape)
                H_Sorted = np.sort(np.concatenate([H1.copy(), H2.copy(), H3.copy(), H4.copy()]))

                L[n][m] = H_Sorted[0:SelectedNumLayers]

        # Normalizing each layer, we only consider SelectedNumLayers number of layers
        L_hat = np.zeros([non_pad_0, non_pad_1, SelectedNumLayers])
        for i in range(SelectedNumLayers):
            L_hat[:, :, i] = self.mat_2_gray(L[:, :, i])

            # Maxpooling
        max_pooled = np.amax(L_hat, 2)  # across dimension 3   since 0,1,2
        if debug == True:
            f = plt.figure()
            plt.title("Max_pooled")
            plt.imshow(max_pooled)

        # taking local entropy by using a 7 by 7 neighbourhood (Not sure cross check this part)
        # The entropy is used for the detection of flat regions in an image
        ent = self.local_entropy(self.mat_2_gray(max_pooled),mat_size=7)

        if debug == True:
            f = plt.figure()
            plt.title("Entropy")
            plt.imshow(ent)

        D = np.multiply(ent,self.mat_2_gray(max_pooled))

        if debug == True:
            f = plt.figure()
            plt.title("D")
            plt.imshow(D)

        # Moving the kernel  over the image and using the filter fspecial
        joint_image = scipy.ndimage.correlate(self.mat_2_gray(GrayImg), self.fspecial_gauss2D([3, 3], 1), mode='nearest')    #.transpose()

        if debug == True:
            f = plt.figure()
            plt.title("Joint")
            plt.imshow(joint_image)

        # imfilter(mat_2_gray(InputImgGray), fspecial_gauss2D([3, 3], 1))
        # FinalMap = self.RF(D, joint_image=joint_image, sigma_s=15, sigma_r=0.25, num_iterations=3)
        FinalMap = self.RF(D, joint_image=joint_image, sigma_s=sig_s, sigma_r=sig_r, num_iterations=num_iter)

        return FinalMap
