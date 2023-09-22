import numpy as np
import pandas as pd
import cv2 as cv
image_url='C:/Users/shubham/Downloads/931c4de4f3cbbeb30a5b65677a174f2980e44805-720x900.jpg'
img=cv.imread(image_url)
cv.imshow("Display window",img)
k=cv.waitKey(0)

# histrogram of the image
# importing library for plotting
from matplotlib import pyplot as plt
  

# gray scale image
grayimg=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("gray image",grayimg)
  
# find frequency of pixels in range 0-255
histr = cv.calcHist([grayimg],[0],None,[256],[0,256])
  
# show the plotting graph of an image
plt.plot(histr)
plt.show()

# normalize histgram
# hist_img1 = cv.calcHist([img], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
normalize=cv.normalize(histr, histr, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
# normalizehist=histr[i]/256*2


plt.plot(normalize)
plt.show()


