import make_d_alt
import numpy as np
import cv2
import math

def D_transpose(img):
    p, n = np.shape(img)
    m = math.floor(p / 2)

    g = np.reshape(img, (p * n, 1), order='F')
    Gx = np.reshape(g[0:m * n], (m, n), order='F')
    Gy = np.reshape(g[(m * n) : p * n], (m, n), order='F')

    Dyi = make_d_alt.make_d_alt(m)
    Dy = -Dyi
    Dxi = make_d_alt.make_d_alt(n)
    Dx = Dxi[0:n, 0: n]
    Dx[0:n, 0] = Dx[0:n, 1] + Dxi[0:n, n]

    altGy = np.zeros((m + 1, n), complex)
    altGy[1:m + 1, 0:n] = Gy
    altGy[0, 1:n] = Gy[m - 1, 0 : n - 1]
    altGy[0, 0] = Gy[m - 1, n - 1]

    delGy = Dy @ altGy
    delGx = Gx @ Dx
    delG = delGx + delGy

    return delG