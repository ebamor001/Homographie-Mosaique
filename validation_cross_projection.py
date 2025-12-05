import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

from homography_cross_projection import homography_cross_projection_two_passes

# 1) Chargement de l'image
I = imread("campagne.jpg")
if I.ndim == 3:
    I = I.mean(axis=2)

# ---- Sélection Quad 1 ----
plt.figure()
plt.imshow(I, cmap='gray')
plt.title("Sélectionner Quad 1 (4 points dans l'ordre)")
pts1 = plt.ginput(4)
plt.close()

x1 = np.array([p[0] for p in pts1])
y1 = np.array([p[1] for p in pts1])

# ---- Sélection Quad 2 ----
plt.figure()
plt.imshow(I, cmap='gray')
plt.title("Sélectionner Quad 2 (4 points dans le même ordre)")
pts2 = plt.ginput(4)
plt.close()

x2 = np.array([p[0] for p in pts2])
y2 = np.array([p[1] for p in pts2])

# ---- Paramètres ----
w1 = h1 = 200
w2 = h2 = 200

# ---- Projections croisées ----
I_swap = homography_cross_projection_two_passes(I, x1, y1, x2, y2, w1, h1, w2, h2)

# ---- Affichage ----
plt.figure()
plt.imshow(I_swap, cmap='gray')
plt.title("Résultat des projections croisées (2 passes)")
plt.axis('off')
plt.show()
