import D_transpose
import T_denom
import numpy as np
import cv2
import math

def T_solver(img, mu, G, U):
    X = G - U

    D_X = D_transpose.D_transpose(X)

    T_num = 2 * img + mu * D_X
    Tn = np.fft.fftshift(np.fft.fft2(T_num))

    size = np.shape(img)
    Td = T_denom.T_denom(size[0], size[1], mu)

    Tnd = np.divide(Tn, Td)

    T = np.fft.ifft2(np.fft.ifftshift(Tnd))

    return T