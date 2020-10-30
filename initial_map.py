import numpy as np
import cv2
import math

def initial_map(img):
    y = np.shape(img)
    a = np.empty(shape = [y[0], y[1]], dtype = np.float)

    b, g, r = cv2.split(img)

    for i in range(len(b)):
        for j in range(len(b[0])):
            if max(b[i][j], max(g[i][j], r[i][j])) == 0:
                a[i][j] = 0.000001

            else:
                a[i][j] = max(b[i][j], max(g[i][j], r[i][j]))

    return a