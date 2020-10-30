import make_d_alt
import initial_map
import D
import make_weight_matrix
import D_transpose
import T_denom
import T_solver
import exact_solver
import numpy as np
import bm3d
import cv2
import math


inp_img = cv2.imread(r'C:\\Users\\adity\\Downloads\\lamp.bmp', 1)

inp_img = cv2.normalize(inp_img.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

ini_map = initial_map.initial_map(inp_img)

ref_map = exact_solver.exact_solver(ini_map, 0.15, 0.8, 1.07)

abs_ref_map = np.absolute(ref_map)

abs_ref_map = np.power(abs_ref_map, 0.8)

R = (np.copy(inp_img))
R = R.astype('float64')
R = R


R[:, :, 0] = np.divide(R[:, :, 0], abs_ref_map)
R[:, :, 1] = np.divide(R[:, :, 1], abs_ref_map)
R[:, :, 2] = np.divide(R[:, :, 2], abs_ref_map)


denoised_image = bm3d.bm3d(R, sigma_psd = 0.05, stage_arg = bm3d.BM3DStages.HARD_THRESHOLDING)
cv2.imwrite('final.bmp', denoised_image*255)
cv2.imshow('win', denoised_image)

cv2.waitKey(0)