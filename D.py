import numpy as np
import cv2
import math
import make_d_alt

def D(img):
    m, n = np.shape(img)

    Dy = make_d_alt.make_d_alt(m)
    Dxt = make_d_alt.make_d_alt(n)
    Dx = np.transpose(Dxt)

    altTy = np.zeros((m + 1, n), complex)
    altTy[0:m, 0: n] = img
    altTy[m, 0: n - 1] = img[0, 1: n]
    altTy[m, n - 1] = img[0, 0]

    delTy = Dy @ altTy
    
    altTx = np.zeros((m, n + 1), complex)
    altTx[0: m, 0: n] = img
    altTx[0: m, n] = img[0: m, 0]

    delTx = altTx @ Dx

    dtx = np.reshape(delTx, (m * n, 1), order='F')
    dty = np.reshape(delTy, (m * n, 1), order='F')

    dt = np.concatenate((dtx, dty))
    dt = np.reshape(dt, (2 * m, n), order= 'F')
    return dt