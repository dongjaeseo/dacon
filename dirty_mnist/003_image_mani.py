import cv2
import numpy as np

img = cv2.imread('../dacon_data/dirty/train/00000.png', cv2.IMREAD_UNCHANGED)
img[img<255] = 0

# blur = cv2.GaussianBlur(img,(5,5),0)
blur2 = cv2.bilateralFilter(img,9,75,75)
blur2 = cv2.dilate(blur2,(3,3), iterations=2)
kernel = np.ones((4,4),np.float32)/25
blur2 = cv2.filter2D(blur2,-1,kernel)
blur2 = blur2/163
# blur2 = blur2/163*255
cv2.imwrite('../dacon_data/clean/train/')