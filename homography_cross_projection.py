import numpy as np

from homography_estimate import homography_estimate
from homography_apply import homography_apply
from scipy.ndimage import map_coordinates
from homography_projection import homography_projection
from homography_extraction import homography_extraction



def homography_cross_projection_two_passes(I, x1, y1, x2, y2, w1, h1, w2, h2):
    # 1) Extraction du premier contenu  
    rect1 = homography_extraction(I, x1, y1, w1, h1)

    # 2) Extraction du second contenu
    rect2 = homography_extraction(I, x2, y2, w2, h2)

    # 3) Projection rect1 -> quad2
    I_swap = homography_projection(rect1, I.copy(), x2, y2)

    # 4) Projection rect2 -> quad1
    I_swap = homography_projection(rect2, I_swap, x1, y1)

    return I_swap


def homography_cross_projection_single_pass(I, x1, y1, x2, y2, S=200):
    """
    Projections croisées via un carré virtuel intermédiaire.
    S : taille du carré virtuel (par défaut 200x200)
    """

    # Carré virtuel cible
    xc = np.array([0, S-1, S-1, 0])
    yc = np.array([0, 0, S-1, S-1])

    # 1) Extraire quad1 → carré virtuel
    H1_c = homography_estimate(x1, y1, xc, yc)
    grid_x, grid_y = np.meshgrid(np.arange(S), np.arange(S))
    gx, gy = grid_x.flatten(), grid_y.flatten()
    xs, ys = homography_apply(H1_c, gx, gy)
    square1 = map_coordinates(I, [ys, xs], order=1).reshape(S, S)

    # 2) Extraire quad2 → carré virtuel
    H2_c = homography_estimate(x2, y2, xc, yc)
    xs, ys = homography_apply(H2_c, gx, gy)
    square2 = map_coordinates(I, [ys, xs], order=1).reshape(S, S)

    # 3) Projeter square1 → quad2
    I_swap = homography_projection(square1, I.copy(), x2, y2)

    # 4) Projeter square2 → quad1
    I_swap = homography_projection(square2, I_swap, x1, y1)

    return I_swap
