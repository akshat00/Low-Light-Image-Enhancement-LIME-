import make_weight_matrix
import G_solver
import T_solver
import D
import numpy as np
import cv2
import math

def exact_solver(img, alpha, mu0, rho):
    size = np.shape(img)
    iterations = 50

    Z = np.zeros((2 * size[0], size[1]))
    G = Z
    k = 0
    mu = mu0

    W = make_weight_matrix.make_weight_matrix(img, 5)

    while k < iterations:
        U = Z / mu
        A = alpha * W / mu
        T = T_solver.T_solver(img, mu, G, U)
        delT = D.D(T)
        G = G_solver.G_solver(A, (delT + U))
        B = delT - G
        Z = mu * (B + U)
        mu = mu * rho
        k = k + 1

    return T