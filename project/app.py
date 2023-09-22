import numpy as np
import pandas as pd
import cv2 as cv
image_url='C:/Users/shubham/Downloads/931c4de4f3cbbeb30a5b65677a174f2980e44805-720x900.jpg'
img=cv.imread(image_url)
# cv.imshow("Display window",img)
# k=cv.waitKey(0)

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


# cummaltive sum
# Calculate cumulative sum
hist_cumsum = histr.cumsum()
hist_cumsum_sq = (histr * np.arange(256)).cumsum()


# size of the image
total_pixels = grayimg.size

best_threshold = 0
max_variance = 0



for threshold in range(1, 256):
    # Calculate the weight and mean of the background
    w_background = hist_cumsum[threshold - 1] / total_pixels
    mean_background = hist_cumsum_sq[threshold - 1] / (w_background * total_pixels)
    
    # Calculate the weight and mean of the foreground
    w_foreground = (hist_cumsum[-1] - hist_cumsum[threshold - 1]) / total_pixels
    mean_foreground = (hist_cumsum_sq[-1] - hist_cumsum_sq[threshold - 1]) / (w_foreground * total_pixels)
    
    # Calculate between-class variance
    between_class_variance = w_background * w_foreground * (mean_background - mean_foreground) ** 2
    
    # Check if the current threshold yields a higher variance
    if between_class_variance > max_variance:
        max_variance = between_class_variance
        best_threshold = threshold

# Apply the best threshold

# Apply the best threshold
_, thresholded_image = cv.threshold(grayimg, best_threshold, 255, cv.THRESH_OTSU)


# Display the original and thresholded images
plt.subplot(1, 2, 1)
plt.imshow(grayimg, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(thresholded_image, cmap='gray')
plt.title('Thresholded Image')
plt.axis('off')

plt.show()


# print("Image shape:", grayimg.shape)
# print("Minimum pixel value:", np.min(grayimg))
# print("Maximum pixel value:", np.max(grayimg))


# print("Best Threshold:", best_threshold)


# plt.hist(grayimg.ravel(), bins=256, range=(0, 256), density=True)
# plt.title("Histogram")
# plt.show()
