import cv2 
import numpy as np
#[131, 218, 311, 549]
img = cv2.imread("dog.jpg")
blur = cv2.GaussianBlur(img, (105, 105), 0)
dog_portion = img[218:549,131:311]
blur[218:549,131:311] = dog_portion
cv2.imshow("image",blur)

cv2.waitKey(0)
cv2.destroyAllWindows()

