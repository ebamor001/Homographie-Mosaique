import numpy as np
from homography_estimate import homography_estimate
from homography_apply import homography_apply
from scipy.ndimage import map_coordinates


def homography_extraction(I1, x, y, w, h):
    #initialiser la destination 
    xd = np.array([0, w-1, w-1, 0])
    yd = np.array([0, 0, h-1, h-1])
    
    #estimer l'homographie
    H = homography_estimate(xd,yd,x,y)
    
    #crée la matrice de destination
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    #transforme matrice h,w en 1,h*w
    xx_flat = xx.ravel()
    yy_flat = yy.ravel()
    
    #appliquer l'homographie pour les pts 
    xs_flat, ys_flat = homography_apply(H, xx_flat, yy_flat)
    
    # rendre en 2,h*w
    coords = np.vstack([ys_flat, xs_flat])
    

    # Interpolation bilinéaire 
    I2_flat = map_coordinates(I1, coords, order=1, mode='constant', cval=0.0)

    # Remise en forme
    I2 = I2_flat.reshape(h, w).astype(I1.dtype)

    
    return I2


