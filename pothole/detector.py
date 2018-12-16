import pandas as pd
import numpy as np
import cv2

# Load in the original image
image = cv2.imread('images/p6.jpeg')
# Transform the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Blur the image
blur = cv2.GaussianBlur(gray,(7,7),0)
kernel = np.ones((2,2),np.uint8)
# Dilate the image 10 times
dilation = cv2.dilate(blur, kernel, iterations=10)
# find canny edges of the dilated image
edged = cv2.Canny(dilation, 30, 200)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

cv2.imshow('Dilated Image', dilation)
cv2.waitKey(0)

_, contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image, contours, -1, (0,255,0),3)
cv2.imshow('Image',image)
cv2.waitKey()
cv2.destroyAllWindows() 