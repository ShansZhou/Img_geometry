import numpy as np
import cv2
import Img_geometry as igeo


imgpath = "./data/lena.jpg"

img = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)


# translation
t = [100, 0]
img_trans = igeo.trasnlateImg(src_img=img, vec=t)
cv2.imshow("img_trans", img_trans)

# scaling
s = [0.5, 0.3]
img_scaled = igeo.scalingImg(src_img=img, scalar=s)
cv2.imshow("img_scaled", img_scaled)

# center rotation
deg = 30.0
img_rotated = igeo.rotatingImg(src_img=img, degree=deg)
cv2.imshow("im_rotated", img_rotated)













cv2.waitKey(0)



