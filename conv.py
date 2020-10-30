import numpy as np
import cv2
import math

def conv(u, v):
    npad = len(v) - 1

    full = np.convolve(u, v, 'full')

    first = npad - npad//2

    x = full[first:first+len(u)]

    return x