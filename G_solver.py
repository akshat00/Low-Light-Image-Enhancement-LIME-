import numpy as np
import cv2
import math

def G_solver(A, X):
    m, n = np.shape(A)

    Z = np.zeros((m, n))

    S = np.multiply(np.sign(X), np.maximum(np.absolute(X) - A, Z))

    return S