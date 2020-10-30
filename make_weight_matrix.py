import D
import gaussian_filter
import conv
import numpy as np
import cv2
import math

def make_weight_matrix(img, ker_size):
    size = np.shape(img)
    p = size[0] * size[1]

    D_img = D.D(img)
    D_img_vec = np.reshape(D_img, (2 * p, 1), order='F')

    dtx = D_img_vec[0:p]
    dty = D_img_vec[p: 2*p]

    w_gauss = gaussian_filter.gaussian_filter((ker_size, 1), 2)
    w_gauss = np.reshape(w_gauss, w_gauss.size, order='F')
    dtx = np.reshape(dtx, dtx.size, order='F')
    dty = np.reshape(dty, dty.size, order='F')
    convl_x = conv.conv(dtx, w_gauss)
    convl_y = conv.conv(dty, w_gauss)

    w_x = 1.0 / (np.absolute(convl_x) + 0.0001)
    w_y = 1.0 / (np.absolute(convl_y) + 0.0001)

    W_vec = np.concatenate((w_x, w_y))
    W = np.reshape(W_vec, (2 * size[0], size[1]), order='F')

    return W