import cv2
import numpy as np
import os

newpath = 'clusteredImages' 
if not os.path.exists(newpath):
    os.makedirs(newpath)
    
img = cv2.imread('image1.jpg')
img_array = np.array(img)
Z = img_array.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS, 15, 1)
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,15,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('Output_1',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('clusteredImages/Output_1.jpg',res2)

img = cv2.imread('image2.jpg')
img_array = np.array(img)
Z = img_array.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1)
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('Output_2',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('clusteredImages/Output_2.jpg',res2)

img = cv2.imread('image4.jpg')
img_array = np.array(img)
Z = img_array.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 13, 1.0)
K = 3
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('Output_3',res2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('clusteredImages/Output_3.jpg',res2)

