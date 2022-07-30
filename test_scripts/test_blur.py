import cv2 
import numpy as np
#[131, 218, 311, 549]
img = cv2.imread("dog.jpg")

blurred_img = cv2.GaussianBlur(img, (105, 105), 0)

# mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
# mask = cv2.rectangle(mask, (131, 218), (311,549), (255, 255, 255), -1)
mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
mask = cv2.circle(mask, (258, 258), 100, (255, 255,255), -1)
out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)
print(out.shape)
cv2.imshow("image",out)
cv2.waitKey(0)
cv2.destroyAllWindows()