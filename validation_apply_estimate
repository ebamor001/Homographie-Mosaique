import numpy as np
from homography_estimate import homography_estimate
from homography_apply import homography_apply


# 4 points source (carré)
x1 = np.array([0, 1, 1, 0])
y1 = np.array([0, 0, 1, 1])

# 4 points destination (quadrilatère)
x2 = np.array([10, 40, 35, 12])
y2 = np.array([20, 18, 45, 50])

# 1) Estimation
H = homography_estimate(x1, y1, x2, y2)
print("H =\n", H)

# 2) Application de l'homographie aux points source
x2_pred, y2_pred = homography_apply(H, x1, y1)

print("\nPoints destination attendus :")
print(x2, y2)

print("\nPoints calculés après transformation :")
print(x2_pred, y2_pred)

