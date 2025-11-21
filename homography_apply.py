def homography_apply(H, x1, y1):

    pts1 = np.vstack((x1, y1, np.ones_like(x1)))  

    pts2 = H @ pts1
    # Normalisation 
    pts2 /= pts2[2] 

    x2 = pts2[0]
    y2 = pts2[1]

    return x2, y2
