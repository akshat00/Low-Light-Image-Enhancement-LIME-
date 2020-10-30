import numpy as np
import cv2
import math

def make_d_alt(m):
    a = np.zeros((m, m + 1))

    for i in range(0, m):
        a[i][i] = -1
        a[i][i + 1] = 1

    return a