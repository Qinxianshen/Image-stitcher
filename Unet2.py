import sys
import cv2
import numpy as np

img1 = cv2.imread("./1.jpg")
img2 = cv2.imread("./2.jpg")

img1_m = np.array(img1)
img2_m = np.array(img2)

print(img1_m)

M = cv2.estimateRigidTransform(img1_m,img2_m,True)
print(M)
H_new = np.row_stack((M,np.array([0,0,1])))
print(H_new)

