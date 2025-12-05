import numpy as np
from homography_estimate import homography_estimate
from homography_apply import homography_apply
from scipy.ndimage import map_coordinates

def homography_projection(I_src, I_dst, x, y):
    h, w = I_src.shape

    # Points rectangulaires source
    xs = np.array([0, w-1, w-1, 0])
    ys = np.array([0, 0, h-1, h-1])

    # Homographie source -> destination
    H = homography_estimate(xs, ys, x, y)

    # grille dans l'image source
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()

    # Projection des pixels de I_src dans I_dst
    xd, yd = homography_apply(H, xx_flat, yy_flat)

    coords = np.array([yd, xd])
    values = map_coordinates(I_src, [yy_flat, xx_flat])

    # Seuls les points qui tombent dans I_dst sont gardés
    Hdst, Wdst = I_dst.shape
    mask = (xd >= 0) & (xd < Wdst-1) & (yd >= 0) & (yd < Hdst-1)

    I_dst[yd[mask].astype(int), xd[mask].astype(int)] = values[mask]

    return I_dst
