import numpy as np
import cv2
import math

def T_denom(m, n, mu):
    dxe = np.zeros((m, n))
    dye = np.zeros((m, n))

    dxe[1, 1] = -1
    dxe[1, 2] = 1
    dye[1, 1] = -1
    dye[2, 1] = 1

    dxf = np.fft.fftshift(np.fft.fft2(dxe))
    dxc = np.conj(dxf)
    dx_mod = np.multiply(dxc, dxf)

    dyf = np.fft.fftshift(np.fft.fft2(dye))
    dyc = np.conj(dyf)
    dy_mod = np.multiply(dyc, dyf)

    Td = 2 + mu * (dx_mod + dy_mod)

    return Td