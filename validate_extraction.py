import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from homography_extraction import homography_extraction

img = plt.imread('./background.jpg')

# Convertir en niveaux de gris
if img.ndim == 3:
    img_gray = img.mean(axis=2)
else:
    img_gray = img

plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.title("Cliquez 4 points (dans l'ordre)")
plt.axis('off')

#4 PTS
pts = plt.ginput(4)   
plt.close()

# On sépare les x et y
x = [p[0] for p in pts]
y = [p[1] for p in pts]

print("Points sélectionnés :")
print("x =", x)
print("y =", y)

#extraction
w, h = 300, 400
I2 = homography_extraction(img_gray, x, y, w, h)

plt.figure()
plt.imshow(I2, cmap='gray')
plt.title("Extraction rectifiée")
plt.axis('off')
plt.show()


#afficher les pts sur l'image d'origine
def show_points_on_image(img, x, y):
    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap='gray')
    plt.scatter(x, y, c='red', s=80)


    plt.axis('off')
    plt.show()

show_points_on_image(img_gray, x, y)
